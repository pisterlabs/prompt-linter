"""
Module for AI processing tasks
"""
import datetime
import io
import logging

import openai
import PIL
import replicate
import tiktoken
import trafilatura
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from openai import OpenAIError
from PIL import Image
from sqlalchemy import desc, select
from stability_sdk.client import StabilityInference, process_artifacts_from_answers
from stability_sdk.utils import generation

from edubot import DREAMSTUDIO_KEY, REPLICATE_KEY
from edubot.sql import Bot, Completion, Message, Session, Thread
from edubot.types import CompletionInfo, ImageInfo, MessageInfo

# The limit for GPT-4 is 8192 tokens.
MAX_GPT_TOKENS = 8192
# The maximum number of GPT tokens that chat context can be.
MAX_PROMPT_TOKENS = MAX_GPT_TOKENS - 1192
# The maximum number of GPT tokens that can be used for completion.
MAX_COMPLETION_TOKENS = MAX_GPT_TOKENS - MAX_PROMPT_TOKENS

# Settings for GPT completion generation
GPT_SETTINGS = {
    "model": "gpt-4",
    "temperature": 0.3,
    "max_tokens": MAX_COMPLETION_TOKENS,
}

LLM = ChatOpenAI(**GPT_SETTINGS)

# The maximum allowed size of images in megabytes
MAX_IMAGE_SIZE_MB = 50

# Prompt for GPT to summarise web pages
WEB_SUMMARY_PROMPT = (
    "Your input is scraped text from a website. Your job is to summarise the text and post it to a chatroom.\n"
    "Long-form text includes pages such as news articles and blog posts.\n"
    "If the page doesn't contain long-form text return the phrase 'NO CONTENT' and nothing else.\n"
    "If the page mentions any variation of 'requiring javascript', or 'enable javascript' you should also return 'NO CONTENT' and nothing else.\n"
    "If the page DOES contain long-form text return a brief 2 sentence summary of the text content. "
    "This summary will then be sent to users.\n"
)

logger = logging.getLogger(__name__)

REPLICATE_CLIENT = replicate.Client(api_token=REPLICATE_KEY)


def estimate_tokens(text: str) -> int:
    # Load GPT-4 encoding
    enc = tiktoken.encoding_for_model(GPT_SETTINGS["model"])

    # Turn text into tokens and count them
    token_count = len(enc.encode(text))
    return token_count


class EduBot:
    """
    An AI chatbot which continually improves itself using user feedback.
    """

    def __init__(self, username: str, platform: str, personality: str | list[str]):
        """
        Initialise EduBot with personalised information about the bot.

        :param username: A unique name to identify this bot from others on the same platform.
        :param platform: The platform the bot is running on E.g. 'telegram' 'matrix' 'mastodon'
        :param personality: Instructions/information for the bot to follow when generating responses.
        """
        self.username = username
        self.platform = platform

        if personality is None:
            self.personality = []
        elif type(personality) is str:
            self.personality = [personality]
        else:
            self.personality = personality

        self.__add_bot_to_db()

        # The primary key of the bot in the database
        self.__bot_pk = self.__get_bot(username).id

        # This variable is lazy loaded
        self.stability_client: StabilityInference | None = None

        self.system_messages = [
            f"You are a chatbot named '{self.username}' which is controlled by an open source python"
            f" program called EduBot that is running on a server owned by the Open EdTech"
            f" organisation. On the backend, the edubot program connects to the API of an"
            f" LLM ({GPT_SETTINGS['model']}) which processes prompts and returns responses. "
            f"On the frontend, the Edubot program connects via API to the {self.platform}"
            f" platform to write posts and read the posts of others. "
            f"You, EduBot, are not able to change yourself - all requests to modify your"
            f" code should be directed to Github: https://github.com/openedtech/edubot.",
            "You should pretend you can view images. "
            "Descriptions of images posted to the chat will be saved to the database. "
            "These descriptions are in the form: '*A picture of ____'. "
            "When you spot these descriptions you should pretend you can see the image, using the description. "
            "Do not mention that you cannot see the image, or that you are instead viewing a description of"
            "the image. Just pretend like you can see it.",
            f"The current year is: {datetime.datetime.now().year}",
            f"You use the language model {GPT_SETTINGS['model']}",
            f"Never prefix your messages with '{self.username}:'",
        ]

    def __get_bot(self, username: str) -> Bot | None:
        """
        Returns the Bot of "username" if it exists on this platform otherwise returns None.
        """
        with Session() as session:
            bot = session.execute(
                select(Bot)
                .where(Bot.username == username)
                .where(Bot.platform == self.platform)
            ).fetchone()

            if bot:
                return bot[0]
            else:
                return None

    def __add_bot_to_db(self) -> None:
        """
        Insert this bot into the DB if it isn't already.
        """
        if not self.__get_bot(self.username):
            with Session() as session:
                new_bot = Bot(username=self.username, platform=self.platform)

                session.add(new_bot)
                session.commit()

    def __get_message(self, msg_info: MessageInfo) -> Message | None:
        """
        Get an ORM Message object from the database.
        """
        with Session() as session:
            message = session.execute(
                select(Message)
                .where(Message.username == msg_info["username"])
                .where(Message.message == msg_info["message"])
                .where(Message.time == msg_info["time"])
                .where(Thread.platform == self.platform)
            ).fetchone()
            if message:
                return message[0]
            else:
                return None

    def __get_thread(self, thread_name: str) -> Thread | None:
        """
        Get an ORM Thread object from the database.
        """
        with Session() as session:
            thread = session.execute(
                select(Thread)
                .where(Thread.thread_name == thread_name)
                .where(Thread.platform == self.platform)
            ).fetchone()

            if thread:
                return thread[0]
            else:
                return None

    def __get_completion_from_message(self, msg_info: MessageInfo) -> Completion | None:
        """
        Gets the bots response to a specific message.
        """
        with Session() as session:
            msg = self.__get_message(msg_info)
            completion = session.execute(
                select(Completion)
                .where(Completion.bot == self.__bot_pk)
                .where(Completion.reply_to == msg.id)
            ).fetchone()

            if completion:
                return completion[0]
            else:
                return None

    def __add_completion(self, completion: str, reply_to: MessageInfo) -> None:
        """
        Add a completion to the database.

        :param completion: The text the bot generated.
        :param reply_to: The message the bot was replying to.
        """
        msg_id = self.__get_message(reply_to).id
        with Session() as session:
            new_comp = Completion(
                bot=self.__bot_pk,
                message=completion,
                reply_to=msg_id,
            )
            session.add(new_comp)
            session.commit()

    def __format_context(
        self, context: list[MessageInfo], personality_override: str = None
    ) -> list[SystemMessage | HumanMessage | AIMessage]:
        """
        Formats chat context and system messages into a chronological list of langchain messages.

        :param context: A list of MessageInfo.
        :return: The context as a list of langchain message objects.
        """
        if personality_override:
            # We need to shallow copy 'self.personality' to avoid modifying the original list
            personality = self.personality.copy()
            personality.append(personality_override)
        else:
            personality = self.personality

        langchain_messages: list[SystemMessage | HumanMessage | AIMessage] = []

        # Append the system messages and personality to the chat context
        for i in self.system_messages + personality:
            langchain_messages.append(SystemMessage(content=i))

        # The context is too long if the prompt is longer than MAX_PROMPT_TOKENS
        token_count = 0

        for msg in context:
            if msg["username"] == self.username:
                langchain_messages.append(AIMessage(content=msg["message"]))
            else:
                langchain_messages.append(HumanMessage(content=msg["message"]))

            token_count += estimate_tokens(msg["message"])

        # Remove messages from the context until the prompt is short enough
        while token_count > MAX_PROMPT_TOKENS:
            token_count -= estimate_tokens(langchain_messages.pop(0).content)

        return langchain_messages

    @staticmethod
    def __describe_image(image: Image.Image) -> str:
        """
        Gets an AI generated description of an image.
        """
        if not REPLICATE_KEY:
            raise RuntimeError(
                "Replicate key is not defined, make sure to supply it in the config."
            )

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")

        if image_bytes.tell() / 1048576 > MAX_IMAGE_SIZE_MB:
            logger.info(f"Skipped image because it was too large.")
            return None

        output: str = REPLICATE_CLIENT.run(
            "j-min/clip-caption-reward:de37751f75135f7ebbe62548e27d6740d5155dfefdf6447db35c9865253d7e06",
            input={"image": image_bytes},
        )

        if not output:
            logger.error("Replicate returned an empty response.")
            return None

        return output

    def save_image_to_context(self, image: ImageInfo, thread_name: str) -> str | None:
        """
        Saves an AI generated description of a user-sent image to the database. This allows GPT to understand what images are
         and how to describe them. The maximum image size in MB can be read from the MAX_IMAGE_SIZE_MB constant.

        :param image: An ImageInfo object.
        :param thread_name: A unique identifier for the thread the image was posted in.
        :returns: The description of the image or None if an error occurred.
        """
        image_description = self.__describe_image(image["image"])

        with Session() as session:
            thread = self.__get_thread(thread_name)
            if not thread:
                thread = Thread(thread_name=thread_name, platform=self.platform)
                session.add(thread)
                session.commit()

            message = Message(
                thread=thread.id,
                username=image["username"],
                message=f"*An image of {image_description}",
                time=image["time"],
            )
            session.add(message)

            session.commit()

        return image_description

    def gpt_answer(
        self,
        new_context: list[MessageInfo],
        thread_name: str,
        personality_override: str = None,
    ) -> str | None:
        """
        Use chat context to generate a GPT3 response.

        :param new_context: Chat context as a chronological list of MessageInfo
        :param thread_name: The unique identifier of the thread this context pertains to
        :param personality_override: A custom personality that overrides the default.

        :returns: The response from GPT
        """
        with Session() as session:
            thread = self.__get_thread(thread_name)

            if not thread:
                thread = Thread(thread_name=thread_name, platform=self.platform)

                session.add(thread)
                session.commit()

            # Context in this timeframe that is in the database but not in the new context provided
            # (Usually images)
            existing_context: list[MessageInfo] = []

            for existing_msg in session.scalars(
                select(Message)
                .where(Message.thread == thread.id)
                .where(Message.time > new_context[0]["time"])
            ):
                row_as_msg_info: MessageInfo = {
                    "username": existing_msg.username,
                    "message": existing_msg.message,
                    "time": existing_msg.time,
                }
                if row_as_msg_info not in new_context:
                    existing_context.append(row_as_msg_info)

            # The existing context in this timeframe + the new messages
            new_and_existing_context: list[MessageInfo] = []

            for index, msg in enumerate(new_context):
                # Figure out where to insert the extra context chronologically
                for extra_msg in existing_context:
                    check = extra_msg["time"] < msg["time"]
                    if index > 0:
                        check = (
                            check and extra_msg["time"] > new_context[index - 1]["time"]
                        )

                    if check:
                        new_and_existing_context.append(extra_msg)
                        existing_context.remove(extra_msg)

                new_and_existing_context.append(msg)

                # If the message is already in the database
                if self.__get_message(msg) is not None:
                    continue

                # If the message was written by a bot
                if self.__get_bot(msg["username"]) is not None:
                    continue

                row: dict = msg
                row["thread"] = thread.id

                session.add(Message(**row))

            session.commit()

        # Ensure that all bot completions are included in context, notably image completions.
        complete_context: list[MessageInfo] = []
        for message in new_and_existing_context:
            if self.__get_bot(message["username"]) is not None:
                continue
            complete_context.append(message)

            if completion := self.__get_completion_from_message(message):
                complete_context.append(
                    {
                        "username": self.username,
                        "message": completion.message,
                        "time": message[
                            "time"
                        ],  # Estimate this, it doesn't matter for gpt_context
                    }
                )

        langchain_context = self.__format_context(
            complete_context, personality_override=personality_override
        )

        chat = ChatOpenAI(**GPT_SETTINGS)
        completion = chat(langchain_context).content

        if not completion:
            return None

        # Strip username from completion, sometimes GPT messes this up.
        completion = completion.replace(f"{self.username}:", "").lstrip()

        # Add a new completion to the database using the completion text and the message being replied to
        self.__add_completion(completion, complete_context[-1])

        # Return the completion result back to the integration
        return completion

    def change_completion_score(
        self, offset: int, completion: CompletionInfo, thread_name: str
    ) -> None:
        """
        Change user feedback to a completion.

        :param offset: An integer representing the new positive or negative votes to this reaction.
        :param completion: Information about the completion being reacted to.
        :param thread_name: A unique identifier for the thread the completion resides in.
        """

        # 1.5 mins before the completion was sent
        delta = completion["time"] - datetime.timedelta(minutes=1, seconds=30)

        with Session() as session:
            # This select statement might get the wrong completion if the bot has sent duplicate messages in the same
            #  thread within 1.5 minutes.
            # BUT this isn't really a problem because it's very likely that users have the same reaction to
            #  both of the duplicate messages.
            # TODO: Is there a way to uniquely identify a bot completion? We can't record the time the completion was
            #  sent as we don't know when the integration sends the completion. The integration also can't know for
            #  sure which message a completion was replying to, as messages can be sent while the bot is generating
            #  responses.
            completion_row = session.execute(
                select(Completion)
                .join(Bot)
                .join(Message)
                .join(Thread)
                .where(Completion.message == completion["message"])
                .where(Thread.thread_name == thread_name)
                .where(Bot.id == self.__bot_pk)
                # The message being replied to was sent not more than 1.5 minutes before the completion
                .where(delta < Message.time)
                .where(Message.time < completion["time"])
                .order_by(desc(Completion.id))
            ).fetchone()

            if not completion_row:
                logger.debug(
                    f"Message is not a GPT completion: '{completion['message']}' @ {completion['time']}"
                )
                return

            completion: Completion = completion_row[0]

            completion.score += offset

            session.add(completion)
            session.commit()

            logger.info(f"Completion {completion.id} incremented by {offset}.")

    def generate_image(
        self, prompt: str, reply_to_msg: MessageInfo, thread_name: str
    ) -> Image.Image | None:
        """
        Generate an image using Stability AI's DreamStudio.

        :param prompt: A description of the image that should be generated.
        :param reply_to_msg: The message the bot is replying to.
        :param thread_name: A unique identifier for the thread the message resides in.
        :return: A PIL.Image.Image instance.
        """
        if not DREAMSTUDIO_KEY:
            raise RuntimeError(
                "DreamStudio key is not defined, make sure to supply it in the config."
            )

        # Lazy load client
        if self.stability_client is None:
            verbose = logger.level >= 10
            self.stability_client = StabilityInference(
                key=DREAMSTUDIO_KEY, verbose=verbose
            )

        # Get Answer objects from stability
        answers = self.stability_client.generate(prompt)

        # Convert answer objects into artifacts we can use
        artifacts = process_artifacts_from_answers("", "", answers, write=False)

        image: Image.Image | None = None

        # noinspection PyBroadException
        try:
            for _, artifact in artifacts:
                # Check that the artifact is an Image, not sure why this is necessary.
                # See: https://github.com/Stability-AI/stability-sdk/blob/d8f140f8828022d0ad5635acbd0fecd6f6fc317a/src/stability_sdk/utils.py#L80
                if artifact.type == generation.ARTIFACT_IMAGE:
                    image = PIL.Image.open(io.BytesIO(artifact.binary))
                    break
        # Exception only happens when prompt is inappropriate.
        except Exception:
            return None

        if image is None:
            return None

        with Session() as session:
            thread = self.__get_thread(thread_name)

            if not thread:
                thread = Thread(thread_name=thread_name, platform=self.platform)

                session.add(thread)
                session.commit()

            message = Message(
                username=reply_to_msg["username"],
                message=reply_to_msg["message"],
                time=reply_to_msg["time"],
                thread=thread.id,
            )
            session.add(message)
            session.commit()

        image_description = self.__describe_image(image)
        completion = (
            f"*An image you generated based on the prompt: '{prompt}'.\n"
            f"*Your interpretation of the image is: '{image_description}'."
        )
        self.__add_completion(completion, reply_to_msg)

        return image

    def summarise_url(self, url: str, msg: MessageInfo, thread_name: str) -> str | None:
        """
        Use GPT to summarise the text content of a URL.

        Returns None if the webpage cannot be fetched or doesn't contain long-form text to summarise.

        :param url: A valid url.
        :param msg: The message that triggered this summary request.
        :param thread_name: A unique identifier for the thread the URL was sent in.
        """
        resp = trafilatura.fetch_url(url)

        # If HTTP or network error
        if resp == "" or resp is None:
            return None

        # Convert HTML to Plaintext
        text = trafilatura.extract(resp)

        # If error converting to plaintext
        if text is None:
            return None

        # Ensure text doesn't exceed GPT limits
        while estimate_tokens(text) > MAX_PROMPT_TOKENS:
            text = text[:-100]

        gpt_context = [
            {"role": "system", "content": WEB_SUMMARY_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            completion = openai.ChatCompletion.create(
                messages=gpt_context,
                **GPT_SETTINGS,
            )
        except OpenAIError as e:
            logger.error(f"OpenAI request failed: {e}")
            return None

        completion_text: str = completion["choices"][0]["message"]["content"]

        if "NO CONTENT" in completion_text.upper():
            return

        completion_text = completion_text.strip()

        with Session() as session:
            thread = self.__get_thread(thread_name)
            if not thread:
                thread = Thread(thread_name=thread_name, platform=self.platform)
                session.add(thread)
                session.commit()

            if self.__get_message(msg) is None:
                row: dict = msg
                row["thread"] = thread.id
                session.add(Message(**row))
                session.commit()

            # Ensure URL summaries are added to the DB
            self.__add_completion(completion_text, msg)

            session.commit()

        return completion_text
