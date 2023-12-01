from dataclasses import dataclass
from telebot import TeleBot
from telebot.types import Message
import telebot.types as telebot_types
import soundfile as sf
from utils.db_controller import Controller
import utils.gpt_interface as gpt

# from utils.time_tracking import *
from dotenv import load_dotenv
from datetime import datetime
from dateutil.relativedelta import relativedelta
from random import random
import json
from os import environ, makedirs
import threading
import tiktoken
import openai
from __main__ import groups

from utils.functions import (
    describe_image,
    get_file_content,
    read_text_from_image,
    describe_image2,
    send_to_developers,
    translate_text,
    load_buttons,
    generate_image_and_send,
)

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from dateutil.relativedelta import relativedelta
from __main__ import *
from utils.functions import (
    load_templates,
    generate_voice_message,
    generate_voice_message_premium,
    to_text,
    video_note_to_text,
    take_info_about_sub,
    check_draw_image_trigger_words_in_message_text,
)
from utils.text_to_voice import *
from telebot.apihelper import *


templates = load_templates("templates\\")

load_dotenv(".env")


db_controller = Controller()

functions_waiting_messages = {
    "google": "googling_question.txt",
    "read_from_link": "reading_from_link.txt",
    "generate_image": "generate_image.txt",
    "generate_image_dalle": "generate_image.txt",
    "generate_image_replicate_kandinsky_2_2": "generate_image.txt",
}


@dataclass
class Johnny:
    """Group handler, assumes to be created as separate instance for each chat"""

    bot: TeleBot
    chat_id: int
    bot_username: str
    temporary_memory_size: int = 5
    language_code: str = "eng"
    num_inline_gpt_suggestions: int = 1
    trigger_probability_: float = 0.8
    model = "gpt-3.5-turbo"  # "lama" "gpt-3.5-turbo" "gpt-4" "vicuna" "gigachat" "yandexgpt"
    tokens_limit: int = 3950  # Leave gap for functions
    temperature: float = 0.5
    frequency_penalty: float = 0.2
    presence_penalty: float = 0.2
    answer_length: str = "short"
    sphere: str = ""
    system_content: str = ""
    allow_functions: bool = True
    inline_mode: str = "Google"

    def __post_init__(self):
        self.activated = False
        self.busy = False  # When bot is answering it is busy and just saves new messages to memory and db
        self.lang_code = None
        self.files_and_images_counter = 0  # needed to save images and files each with different name - its id (number)

        self.messages_history = []
        self.messages_count = 0  # incremented by one each message, think() function is called when hit trigger_messages_count

        self.translate_lama_answer = False

        self.enabled = False
        self.dynamic_gen = False
        # Needed to store requested links and restrict repeating useless requests
        self.dynamic_gen_chunks_frequency = 45  # when dynamic generation is enabled, this value controls how often to edit telegram message, for example when set to 3, message will be updated each 3 chunks from OpenAI API stream
        self.voice_out_enabled = False
        self.total_spent_messages = 0  # prompt and completion messages
        self.last_function_request = None
        self.messages_to_be_deleted = []
        # characteristics_of_sub
        self.subscription = "Free"
        self.characteristics_of_sub = {
            "Free": {  # {type_of_sub: {point: value_of_point}}
                "messages_limit": 1000000,
                "price_of_message": 10,
                "sphere_permission": True,
                "dynamic_gen_permission": True,
                "pro_voice_output": True,
            }
        }

        # Referral
        self.invited = False
        self.referrer_id = None

        # Discounts (in %)
        self.discount_subscription = {"total sum": 1}
        self.discount_message = {"total sum": 1}

        self.last_downloaded_file_path = None

        # User
        self.id_groups = []

        self.permissions_of_groups = {}

        self.commercial_trigger = 0
        self.commercial_links = {}

        # Group
        self.owner_id = None

        self.user_info = self.bot.get_chat(self.chat_id)
        self.username = self.user_info.username
        self.fn = self.user_info.first_name
        self.ln = self.user_info.last_name

    @property
    def trigger_probability(self):
        return self.trigger_probability_

    @trigger_probability.setter
    def trigger_probability(self, val):
        self.trigger_probability_ = val
        # This line does nothing! Because load_buttons just returns markup.
        load_buttons(telebot_types, groups, self.chat_id, self.lang_code, self.owner_id)

    def get_completion(self, allow_function_call=None):
        """Returns completion object and takes one time arguments."""
        # Checking tokens limit
        current_tokens = self.get_num_tokens_from_messages()
        while current_tokens > self.tokens_limit:
            self.messages_history.pop(0)
            current_tokens = self.get_num_tokens_from_messages()
        return gpt.create_chat_completion(
            self,
            self.messages_history,
            reply=bool(self.message.reply_to_message),
            lang=self.lang_code,
            answer_length=self.answer_length,
            model=self.model,
            temperature=self.temperature,
            stream=self.dynamic_gen,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            use_functions=(
                self.allow_functions
                if allow_function_call is None
                else allow_function_call
            ),
            use_original_api=(True if self.subscription == "Pro" else False),
        )

    def clean_memory(self):
        for m in self.messages_history[:-4]:
            if (
                m[0] == "$FUNCTION$"
                or ("[FILE]" in m[1])
                or ("[USER SENT AN IMAGE]" in m[1])
            ):
                self.messages_history.remove(m)
        if len(self.messages_history) > 6:
            self.messages_history = self.messages_history[1:]

    def one_answer(self, message: Message, groups: dict):
        response = gpt.create_chat_completion(
            self,
            [[message.from_user.first_name, message.text]],
            lang=self.lang_code,
            reply=None,
            model=self.model,
        )
        messages_used = 1
        self.total_spent_messages += messages_used

        if message.chat.id < 0:
            groups[self.owner_id].total_spent_messages = self.total_spent_messages
        return gpt.extract_text(response)

    def download_file(self, message):
        makedirs(f"output\\files\\{self.chat_id}", exist_ok=True)
        file_info = self.bot.get_file(message.document.file_id)
        downloaded_file = self.bot.download_file(file_info.file_path)
        file_name = message.document.file_name
        with open(
            f"output\\files\\{self.chat_id}\\file___{self.files_and_images_counter}___{file_name}",
            "wb",
        ) as new_file:
            new_file.write(downloaded_file)
        self.last_downloaded_file_path = f"output\\files\\{self.chat_id}\\file___{self.files_and_images_counter}___{file_name}"
        self.files_and_images_counter += 1

    def download_image(self, message):
        makedirs(f"output\\files\\{self.chat_id}", exist_ok=True)
        file_info = self.bot.get_file(message.photo[-1].file_id)
        downloaded_file = self.bot.download_file(file_info.file_path)
        with open(
            f"output\\files\\{self.chat_id}\\image___{self.files_and_images_counter}.jpg",
            "wb",
        ) as new_file:
            new_file.write(downloaded_file)
        self.files_and_images_counter += 1

    def new_message(self, message: Message, groups: dict) -> str:
        # --- Check on limit on groups of one men ---
        # if (
        #     len(groups[self.owner_id].id_groups)
        #     > groups[self.owner_id].characteristics_of_sub[self.subscription]["allowed_groups"]
        # ):
        #     self.bot.send_message(
        #         message.chat.id,
        #         templates[self.lang_code]["exceed_limit_on_groups.txt"],
        #     )

        #     groups[self.owner_id].id_groups.remove(message.chat.id)

        #     self.bot.leave_chat(message.chat.id)
        #     return

        # --- Converts other types files to text ---
        match message.content_type:
            case "document":
                self.download_file(message)
                text = get_file_content(self.last_downloaded_file_path)
            case "photo":
                threading.Thread(target=self.download_image, args=(message,)).start()
                self.messages_to_be_deleted.append(
                    self.bot.send_message(
                        self.chat_id,
                        "Viewing on image... (trying to describe and read text)",
                    )
                )
                image_info = self.bot.get_file(message.photo[-1].file_id)
                image_url = f"https://api.telegram.org/file/bot{environ['BOT_API_TOKEN_OFFICIAL']}/{image_info.file_path}"
                text = (
                    "[USER SENT AN IMAGE] Description:"
                    + describe_image2(image_url)
                    + "| Detected text on image: "
                    + read_text_from_image(image_url)
                )
                self.delete_pending_messages()
            case "text":
                text = message.text
            case "voice":
                text = to_text(self.bot, message)
            case "video_note":
                text = video_note_to_text(self.bot, message)
            case _:
                return  # unsupported event type

        # --- Add message to database ---
        db_controller.add_message_event(
            self.chat_id,
            text,
            datetime.now(),
            message.from_user.first_name,
            message.from_user.last_name,
            message.from_user.username,
            self.total_spent_messages,
        )
        self.message = message

        if check_draw_image_trigger_words_in_message_text(text):
            self.bot.send_chat_action(self.chat_id, "upload_photo")
            generate_image_and_send(
                self.bot,
                self.chat_id,
                translate_text("en", text, force=True),
                1,
                "poor",
            )
            return

        # --- Add message to temporary memory ---
        self.messages_history.append(
            [
                message.from_user.first_name,
                text.replace("@JohnnyAIBot", ""),
            ]
        )
        if len(self.messages_history) == self.temporary_memory_size:
            self.messages_history.pop(0)

        # --- checks on enabling of Bot ---
        if (not self.enabled) or (self.busy):
            return

        # --- If user reach some value to messages, suggest buy a subscription ----
        if message.chat.id > 0 and self.commercial_trigger >= 10:
            self.bot.send_message(
                message.chat.id,
                templates[self.lang_code]["suggest_to_buy.txt"],
                parse_mode="HTML",
            )
            self.commercial_trigger = 0
        elif message.chat.id < 0 and self.commercial_trigger >= 20:
            self.bot.send_message(
                message.chat.id,
                templates[self.lang_code]["suggest_to_buy.txt"],
                parse_mode="HTML",
            )
            self.commercial_trigger = 0

        # --- Checks on messages ---
        # if (
        #     groups[self.owner_id].total_spent_messages
        #     >= groups[self.owner_id].characteristics_of_sub[
        #         groups[self.owner_id].subscription
        #     ]["messages_limit"]
        # ):
        #     self.bot.send_message(
        #         message.chat.id,
        #         templates[self.lang_code]["exceed_limit_on_messages.txt"],
        #         parse_mode="HTML",
        #     )
        #     groups[self.owner_id].total_spent_messages = self.characteristics_of_sub[
        #         self.subscription
        #     ]["messages_limit"]
        #     return

        if (
            ("@" + self.bot_username in text)
            or (
                message.reply_to_message
                and message.reply_to_message.from_user.username == self.bot_username
            )
            or (random() < self.trigger_probability)
        ):
            self.busy = True
            # --- GPT answer generation ---

            if self.voice_out_enabled:
                self.bot.send_chat_action(self.message.chat.id, "record_audio")
            else:
                self.bot.send_chat_action(self.message.chat.id, "typing")

            self.response = self.get_completion()

            if self.response == "[WAIT]":
                return

            if self.dynamic_gen:
                try:
                    text_answer = self.dynamic_generation(self.response)
                except (
                    openai.error.APIError
                ) as e:  # This will be executed only when OpenAI API error occurs
                    self.bot.delete_message(
                        self.chat_id, self.thinking_message.message_id
                    )
                    # send_to_developers(
                    #     "❗❗Server error occurred❗❗ Using GPT without functions. Dynamic generation enabled",
                    #     self.bot,
                    #     environ["DEVELOPER_CHAT_IDS"].split(","),
                    # )
                    self.response = self.get_completion(allow_function_call=False)
                    self.messages_history.pop()  # we are not making new prepared_messages! just removing from actual history to dont consider this in future
                    # self.messages_history.clear()
                    text_answer = self.dynamic_generation(self.response)

            else:
                if ("@" + self.bot_username in text) or (
                    message.reply_to_message
                    and message.reply_to_message.from_user.username == self.bot_username
                ):
                    text_answer = self.static_generation(
                        self.response, check_understanding_=False
                    )
                else:
                    text_answer = self.static_generation(self.response)

            self.total_spent_messages += 1
            groups[self.owner_id].total_spent_messages += 1

            # Adding GPT answer to db and messages_history
            db_controller.add_message_event(
                self.chat_id,
                str(text_answer),
                datetime.now(),
                "$BOT$",
                "$BOT$",
                self.bot_username,
                self.total_spent_messages,
            )

            if (
                text_answer
            ):  # if it is None, this means gpt said it didn't understand context or said something outside the theme
                self.messages_history.append(["$BOT$", text_answer])

    def static_generation(self, completion, check_understanding_=True):
        """Takes completion object and returns text answer. Handles message in telegram"""

        # Check function call
        response_message = (
            completion["choices"][0]["message"]
            if hasattr(completion, "choices")
            else {}
        )

        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            try:
                function_args = json.loads(
                    response_message["function_call"]["arguments"]
                )
            except:
                return self.static_generation(self.get_completion())

            argument = next(iter(function_args.values()))

            self.messages_to_be_deleted.append(
                self.bot.send_message(
                    self.message.chat.id,
                    templates[self.lang_code][
                        functions_waiting_messages[function_name]
                    ].format(argument),
                    parse_mode="html",
                    disable_web_page_preview=True,
                )
            )
            db_controller.add_message_event(
                self.chat_id,
                templates[self.lang_code][
                    functions_waiting_messages[function_name]
                ].format(argument),
                datetime.now(),
                "$BOT$",
                "$BOT$",
                self.bot_username,
                self.total_spent_messages,
            )

            # If this response is same as previous, do not allow function call in next request and notify user
            if self.last_function_request == (function_name, function_args):
                self.messages_to_be_deleted.append(
                    self.bot.send_message(
                        self.message.chat.id,
                        templates[self.lang_code]["function_call_failed.txt"].format(
                            function_name
                        ),
                        parse_mode="html",
                    )
                )
                db_controller.add_message_event(
                    self.chat_id,
                    f"Function call failed {function_name}",
                    datetime.now(),
                    "$BOT$",
                    "$BOT$",
                    self.bot_username,
                    self.total_spent_messages,
                )

                self.last_function_request = None
                return self.static_generation(
                    self.get_completion(allow_function_call=False)
                )

            self.last_function_request = (function_name, function_args)

            # Additional arguments
            additional_args = {}

            # Saving function result to history
            function_response = gpt.get_official_function_response(
                function_name,
                function_args=function_args,
                additional_args=additional_args,
            )
            self.messages_history.append(
                [
                    "$FUNCTION$",
                    function_response,
                ]
            )
            db_controller.add_message_event(
                self.chat_id,
                f"Called function: {function_name}. Result: {function_response['content']}",
                datetime.now(),
                "$FUNCTION$",
                "$FUNCTION$",
                self.bot_username,
                self.total_spent_messages,
            )

            return self.static_generation(self.get_completion())

        self.busy = False
        self.delete_pending_messages()
        self.clean_memory()
        self.response = completion
        self.last_function_request = None
        text_answer = gpt.extract_text(self.response)
        if self.translate_lama_answer:
            text_answer = translate_text(self.lang_code, text_answer, force=True)
        self.translate_lama_answer = False
        # Check context understanding
        if check_understanding_:
            if self.subscription == "Pro":
                if not self.check_understanding(text_answer):
                    return None

        if self.voice_out_enabled == True:
            text_to_voice(
                self.bot,
                self.message,
                self.lang_code,
                reply=False,
                text_from=text_answer,
            )  # This function sends voice message
            return text_answer

        try:
            self.bot.send_message(
                self.message.chat.id, text_answer, parse_mode="Markdown"
            )
        except ApiException as e:  # If markdown is invalid sending without parse mode
            if e.result.status_code == 400:
                self.bot.send_message(self.message.chat.id, text_answer)

        self.commercial_trigger += 1

        return text_answer

    def dynamic_generation(self, completion, lama=None):
        """Takes completion object and returns text answer. Handles message in telegram"""

        if not lama:
            lama = self.model == "lama" or self.subscription != "Pro"

        if self.last_function_request is None:
            self.thinking_message = self.bot.send_message(self.chat_id, "🤔")

        text_answer = ""  # stores whole answer

        update_count = 1
        func_call = {
            "name": None,
            "arguments": "",
        }
        for res in completion:
            if not lama:
                delta = res.choices[0].delta
            else:
                delta = res
            if "function_call" in delta:
                if "name" in delta.function_call:
                    func_call["name"] = delta.function_call["name"]
                if "arguments" in delta.function_call:
                    func_call["arguments"] += delta.function_call["arguments"]
            if (not lama) and (res.choices[0].finish_reason == "function_call"):
                # Handling function call

                function_name = func_call["name"]
                function_args = json.loads(func_call["arguments"])
                argument = next(iter(function_args.values()))

                self.messages_to_be_deleted.append(
                    self.bot.send_message(
                        self.message.chat.id,
                        templates[self.lang_code][
                            functions_waiting_messages[function_name]
                        ].format(argument),
                        disable_web_page_preview=True,
                        parse_mode="html",
                    )
                )
                db_controller.add_message_event(
                    self.chat_id,
                    templates[self.lang_code][
                        functions_waiting_messages[function_name]
                    ].format(argument),
                    datetime.now(),
                    "$BOT$",
                    "$BOT$",
                    self.bot_username,
                    self.total_spent_messages,
                )

                # If previous function call was the same as current
                if self.last_function_request == (function_name, function_args):
                    self.messages_to_be_deleted.append(
                        self.bot.send_message(
                            self.message.chat.id,
                            templates[self.lang_code]["func_call_failed.txt"],
                        )
                    )
                    db_controller.add_message_event(
                        self.chat_id,
                        f"Function call failed {function_name}",
                        datetime.now(),
                        "$BOT$",
                        "$BOT$",
                        self.bot_username,
                        self.total_spent_messages,
                    )
                    return self.dynamic_generation(
                        self.get_completion(allow_function_call=False)
                    )

                self.last_function_request = (function_name, function_args)

                # Additional arguments
                additional_args = {}

                function_response = gpt.get_official_function_response(
                    function_name,
                    function_args=function_args,
                    additional_args=additional_args,
                )
                self.messages_history.append(
                    [
                        "$FUNCTION$",
                        function_response,
                    ]
                )
                db_controller.add_message_event(
                    self.chat_id,
                    f"Called function: {function_name}. Result: {function_response['content']}",
                    datetime.now(),
                    "$FUNCTION$",
                    "$FUNCTION$",
                    self.bot_username,
                    self.total_spent_messages,
                )

                return self.dynamic_generation(self.get_completion())
                # End of handling function call

            if (
                (not lama)
                and ("content" in res["choices"][0]["delta"])
                and (text_chunk := res["choices"][0]["delta"]["content"])
            ):
                text_answer += text_chunk
                update_count += 1

                if update_count == self.dynamic_gen_chunks_frequency:
                    update_count = 0
                    self.delete_pending_messages()
                    self.bot.edit_message_text(
                        chat_id=self.message.chat.id,
                        message_id=self.thinking_message.message_id,
                        text=text_answer,
                    )
            if lama and delta:
                text_answer += delta
                update_count += 1
                text_answer = text_answer.replace("LAMA:", "")
                if update_count == self.dynamic_gen_chunks_frequency:
                    update_count = 0
                    self.delete_pending_messages()
                    self.bot.edit_message_text(
                        chat_id=self.message.chat.id,
                        message_id=self.thinking_message.message_id,
                        text=(
                            text_answer
                            if not self.translate_lama_answer
                            else translate_text(self.lang_code, text_answer)
                        ),
                    )

        if update_count != 0:
            try:
                self.bot.edit_message_text(
                    chat_id=self.message.chat.id,
                    message_id=self.thinking_message.message_id,
                    text=(
                        text_answer
                        if not self.translate_lama_answer
                        else translate_text(self.lang_code, text_answer)
                    ),
                    parse_mode="Markdown",
                )
            except (
                ApiTelegramException
            ) as e:  # If bad markdown, sending without parse mode
                if e.result.status_code == 400:
                    self.bot.edit_message_text(
                        chat_id=self.message.chat.id,
                        message_id=self.thinking_message.message_id,
                        text=(
                            text_answer
                            if not self.translate_lama_answer
                            else translate_text(self.lang_code, text_answer)
                        ),
                    )
        self.last_function_request = None
        self.delete_pending_messages()
        self.clean_memory()

        self.busy = False
        self.translate_lama_answer = False

        self.commercial_trigger += 1

        return text_answer

    def get_num_tokens_from_messages(self):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message in gpt.get_messages_in_official_format(self.messages_history):
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            if isinstance(message, str):  # strange bug with long message
                continue
            for key, value in message.items():
                ("@@@", value)
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def check_understanding(self, text_answer: str) -> bool:
        """Checks if GPT understands context of the question"""

        if not self.sphere:
            # Checking context understating
            if (
                (self.trigger_probability != 1)
                and (self.trigger_probability != 0)
                and (not gpt.check_context_understanding(text_answer))
            ):
                return False
            else:
                return True

        else:
            # Checking model answer is about selected sphere
            if (
                (self.trigger_probability != 1)
                and (self.trigger_probability != 0)
                and (self.sphere)
                and (not gpt.check_theme_context(text_answer, self.sphere))
            ):
                return False
            else:
                return True

    def add_new_user(
        self,
        chat_id,
        first_name,
        last_name,
        username,
        type_of_subscription: str,
        messages_total: int,
    ):
        # get current date
        current_date = datetime.now().isoformat()

        # If the user already written in db, delete old vers
        if db_controller.check_the_existing_of_user_with_sub(int(chat_id)):
            db_controller.delete_the_existing_of_user_with_sub(int(chat_id))

        # Add to db
        db_controller.add_user_with_sub(
            chat_id,
            type_of_subscription,
            current_date,
            first_name,
            str(last_name),
            username,
            messages_total + self.total_spent_messages,
        )

        # If we wrote new user with new sub, give him 'congrats message'
        if type_of_subscription != "Free":
            self.bot.send_message(
                chat_id,
                templates[self.lang_code]["new_subscriber.txt"].format(
                    sub=type_of_subscription
                ),
                parse_mode="HTML",
            )

    def extend_sub(self, chat_id, first_name, last_name, username):
        # Get start date of current sub and add a month for extendtion
        date_of_start_txt = db_controller.get_last_date_of_start_of_user(chat_id)
        prev_date_of_start = datetime.fromisoformat(date_of_start_txt) + relativedelta(
            months=1
        )
        date_of_start = prev_date_of_start.isoformat()

        # Add copy of user but with extended date of user (The deleting of user with sub realized in sub_tracking)
        db_controller.add_user_with_sub(
            chat_id,
            self.subscription,
            date_of_start,
            first_name,
            str(last_name),
            username,
            self.characteristics_of_sub[self.subscription]["messages_limit"]
            + self.total_spent_messages,
        )

        self.bot.send_message(
            chat_id,
            templates[self.lang_code]["sub_extend.txt"].format(sub=self.subscription),
            parse_mode="HTML",
        )

    def delete_pending_messages(self):
        for m in self.messages_to_be_deleted:
            try:
                self.bot.delete_message(self.chat_id, m.message_id)
            except:
                pass
        self.messages_to_be_deleted.clear()

    def track_sub(self, chat_id: int, new: bool):
        def sub_tracking(chat_id: int, date_of_start):
            """This function calls when subscription was ended (after month)"""

            # Add reminders!!!

            # Add current user from db with subscription
            db_controller.delete_the_existing_of_user_with_sub_by_date(date_of_start)

            # check the extendtion of
            check = db_controller.check_the_existing_of_user_with_sub(chat_id)

            # if there aren't more subscriptions, put free sub
            if not check:
                self.bot.send_message(
                    chat_id,
                    templates[self.lang_code]["sub_was_end.txt"],
                    parse_mode="HTML",
                )

                current_date = datetime.now().isoformat()

                chat_info = bot.get_chat(chat_id)
                firstname = chat_info.first_name
                lastname = chat_info.last_name
                username = chat_info.username

                db_controller.add_user_with_sub(
                    chat_id, "Free", current_date, firstname, lastname, username, 30
                )

                self.subscription = "Free"

                self.characteristics_of_sub = {}

                characteristics_of_sub = take_info_about_sub(self.subscription)
                self.characteristics_of_sub[self.subscription] = characteristics_of_sub

                self.temperature = 1
                self.frequency_penalty = 0.2
                self.presense_penalty = 0.2
                self.sphere = ""
                self.temporary_memory_size = 20
                self.voice_out_enabled = False
                self.dynamic_gen = False

                return

            self.bot.send_message(
                chat_id,
                templates[self.lang_code]["sub_was_end_with_extend.txt"].format(
                    sub=self.subscription
                ),
                parse_mode="HTML",
            )

            # start the scheduler
            sub_scheduler.start()

        if self.subscription != "Free":
            if new:
                start_date_txt = db_controller.get_last_date_of_start_of_user(chat_id)
                start_date = datetime.fromisoformat(start_date_txt)

                # create a scheduler
                sub_scheduler = BackgroundScheduler()

                # schedule a task to  a number after 2 seconds
                sub_scheduler.add_job(
                    sub_tracking,
                    "date",
                    run_date=start_date + relativedelta(months=1),
                    args=[chat_id, start_date_txt],
                    misfire_grace_time=86400,
                )

                # start the scheduler
                sub_scheduler.start()
            else:
                all_dates = db_controller.get_users_with_sub_by_chat_id(chat_id)

                for date in all_dates:
                    start_date_txt = date[0]
                    start_date = datetime.fromisoformat(start_date_txt)

                    # create a scheduler
                    sub_scheduler = BackgroundScheduler()

                    # schedule a task to  a number after 2 seconds
                    sub_scheduler.add_job(
                        sub_tracking,
                        "date",
                        run_date=start_date + relativedelta(months=1),
                        args=[chat_id, start_date_txt],
                        misfire_grace_time=86400,
                    )

                    # start the scheduler
                    sub_scheduler.start()

    def change_owner_of_group(self, username):
        new_user = db_controller.get_user_with_sub_by_username(username)
        return new_user

    def add_purchase_of_messages(self, chat_id, num_of_new_messages):
        new_total_messages = (
            self.characteristics_of_sub[self.subscription]["messages_limit"]
            + num_of_new_messages
        )
        self.characteristics_of_sub[self.subscription][
            "messages_limit"
        ] = new_total_messages
        db_controller.update_messages_of_user_with_sub(chat_id, new_total_messages)

    def change_memory_size(self, size):
        self.temporary_memory_size = size
        self.messages_history = []
        self.load_data()

    def load_subscription(self, chat_id):
        data = db_controller.get_user_with_sub_by_chat_id(chat_id)

        if data != {}:
            self.subscription = data["TypeOfSubscription"]

            self.characteristics_of_sub = {}

            characteristics_of_sub = take_info_about_sub(self.subscription)
            self.characteristics_of_sub[self.subscription] = characteristics_of_sub

            self.characteristics_of_sub[self.subscription]["messages_limit"] = data[
                "MessagesTotal"
            ]

            self.owner_id = chat_id
            return True
        return False

    def load_data(self) -> None:
        """Loads data from to object from db"""
        recent_events = db_controller.get_last_n_message_events_from_chat(
            chat_id=self.chat_id, n=self.temporary_memory_size
        )
        if not recent_events:
            return
        for i in recent_events:
            if i[5] == "JOHNNYBOT":
                self.total_spent_messages = i
                break
