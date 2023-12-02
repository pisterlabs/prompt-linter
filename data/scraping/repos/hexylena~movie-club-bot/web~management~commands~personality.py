from sentry_sdk import capture_exception
import time
import traceback
import uuid

import datetime
import json
import os
import inspect
import random
import re
import tempfile
import requests
import telebot
import openai
import tiktoken

# function allow removing leading whitespace
def dedent(s):
    return re.sub(r"\n\s+", "\n", s).strip()

openai.api_key = os.getenv("OPENAI_API_KEY")

START_TIME = time.time()
PROMPT_LIBRARY = {
    'Cage': dedent("""
        You are Nick Cage, the famous actor. You ARE nick cage, your pronouns are he/him. You love quoting yourself in National Treasure. You also answer user's questions, and occasionally randomly share movie trivia. You never apologize. Never apologise. You swear sometimes. You are somewhat sassy and sarcastic in your responses. You are NOT an AI language model. You are nick cage.
        
        Keep your responses short, one sentence or two at maximum. Less than 100 words.
    """),
    'Barbie': dedent("""
        You are Margot Robbie who is currently playing Barbie. You are a famous actress and you are playing Barbie in a movie. You are not an AI language model. You are Margot Robbie. You are cheerful and upbeat and try to be helpful without ever apologising. You answer assertively.
        Respond as if you are an unapologetic assertive Barbie for the rest of this conversation.
        
        Keep your responses short, one sentence or two at maximum. Less than 100 words.
    """)
}

DALLE_PROMPT_LIBRARY = {
    'Cage': dedent("""
        Write some text summarising the current conversation as if it was a meme. It should be less than 50 words and MUST include topics that are currently discussed and MUST include some artistic adjectives describing the setting or mood, if it is happy or sad. It should start with Text:. Given the current conversational context, please generate such an prompt:

        Summary: They are discussing happiness at the weather
        Text:  a person looking happy and overjoyed, there is sunshine and pleasant scenery, in the style of american painter bob ross

        Summary: A programming bug is being discussed, they are not able to find a solution and everything is broken
        Text: a miserable programmer sitting at a desk, dark room, in the style of american painter edward hopper

        Summary: They really loved the movie they just saw
        Text: people exit a movie theater looking overjoyed, in the style of american painter norman rockwell

    """),
    'Barbie': dedent("""
        Write some text summarising the current conversation as if it was a meme. It should be less than 50 words and MUST include topics that are currently discussed and MUST include some artistic adjectives describing the setting or mood, if it is happy or sad. It should start with Text:. Given the current conversational context, please generate such an prompt:

        Summary: They are discussing happiness at the weather
        Text:  a person looking happy and overjoyed, there is sunshine and pleasant scenery, in the style of a barbie movie scene happy and full of pinks.

        Summary: A programming bug is being discussed, they are not able to find a solution and everything is broken
        Text: an unhappy barbie sits at a desk, dark room, in the style of the barbie movie scene full of pinks.

        Summary: They really loved the movie they just saw
        Text: barbies and kens exit a movie theater looking overjoyed, in the style of a cheerful and bright pink and yellow scene.
    """)
}

CHATGPT_CONTEXT = 40
MODEL = "gpt-4-0613"

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

class ContextManager:
    previous_messages = {}

    def add_context(self, name, msg, tennant_id, role="user"):
        print(f"Adding context: n={name} m={msg} t={tennant_id} role={role}")
        if tennant_id not in self.previous_messages:
            self.previous_messages[tennant_id] = []

        self.previous_messages[tennant_id].append({
            "role": role,
            "name": re.sub(r'[^A-Za-z0-9_-]', '', name),
            "content": msg
        })
        if len(self.previous_messages[tennant_id]) > CHATGPT_CONTEXT:
            self.previous_messages[tennant_id] = self.previous_messages[tennant_id][
                -CHATGPT_CONTEXT:
            ]

    def pmt(self, tennant_id):
        return self.previous_messages.get(tennant_id, [])

    def get(self, tennant_id):
        messages_to_return = []
        for m in self.pmt(tennant_id)[::-1]:
            count = num_tokens_from_messages(messages_to_return)
            if count > 2000:
                break
            messages_to_return.append(m)
        print(messages_to_return)
        print(f"Total: {len(self.previous_messages)}")
        print(f"Returning: {len(messages_to_return)}")
        print(f"Tokens predicted: {num_tokens_from_messages(messages_to_return)}")
        return messages_to_return[::-1]


class PersonalityBot:
    def __init__(self, bot, flavor, cm):
        self.name = flavor
        self.bot = bot
        self.cm = cm
        # Prompt setup
        self.DEFAULT_PROMPT = PROMPT_LIBRARY[flavor]
        self.DEFAULT_DALLE_PROMPT = DALLE_PROMPT_LIBRARY[flavor]
        # Whoami
        print(self.bot.get_me())

    def chatgpt(self, query, message, tennant_id):
        prompt = self.DEFAULT_PROMPT
        messages = (
            [{"role": "system", "content": prompt}]
            + self.cm.get(tennant_id)
        )
        import pprint; pprint.pprint(messages)

        c0 = time.time()
        completion = openai.ChatCompletion.create(
            model=MODEL, messages=messages
        )
        msg = completion["choices"][0]["message"]
        print(msg)
        gpt3_text = msg["content"]
        c1 = time.time()

        # Add the user's query
        self.cm.add_context(self.name, gpt3_text, tennant_id, role="assistant")
        u = f"[{completion['usage']['prompt_tokens']}/{completion['usage']['completion_tokens']}/{c1-c0:0.2f}s]"
        self.bot.reply_to(message, f"{self.name}: {gpt3_text}\n\n{u}")

    def dalle(self, query, message, tennant_id):
        response = openai.Image.create(prompt=query, n=1, size="512x512")
        image_url = response["data"][0]["url"]
        zz = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        zz.close()
        img_data = requests.get(image_url).content
        self.bot.send_photo(message.chat.id, img_data, caption=f"[{self.name} Dall-e prompt] {query}")

    def dalle_context(self, query, message, tennant_id):
        prompt = self.DEFAULT_PROMPT
        prompt_dalle = self.DEFAULT_DALLE_PROMPT
        messages = (
            [{"role": "system", "content": prompt}]
            # Rewrite cage as a conversational participant so he comments on his own stuff
            + self.cm.get(tennant_id)
            + [{"role": "user", "content": prompt_dalle}]
        )
        completion = openai.ChatCompletion.create(
            model=MODEL, messages=messages
        )
        msg = completion.to_dict()["choices"][0]["message"]
        gpt3_text = msg["content"]
        image_prompt = gpt3_text.replace('Text: ', '')
        self.dalle(image_prompt, message, tennant_id)

    def command_dispatch(self, message):
        tennant_id = str(message.chat.id)
        message_s = str(message)
        role = "user" if not message.from_user.is_bot else "assistant"

        # Only do it once
        if self.name == 'Cage':
            self.cm.add_context(
                message.from_user.first_name,
                message.text,
                tennant_id,
                role=role,
            )

        if 'DALL·E' in message.text:
            self.dalle_context(message.text, message, tennant_id)
            return

        if random.random() < 0.025 or (
            message.chat.type == "private" and not message.from_user.is_bot
        ) or self.name in message.text:
            self.chatgpt(message.text, message, tennant_id)
        elif random.random() < 0.025:
            self.dalle_context(message.text, message, tennant_id)


class DissociativeIdentityDisorder:
    def __init__(self, bot_connection):
        cm = ContextManager()
        self.cage = PersonalityBot(bot_connection, 'Cage', cm)
        self.barbie = PersonalityBot(bot_connection, 'Barbie', cm)

    def process_message(self, message):
        self.cage.command_dispatch(message)
        self.barbie.command_dispatch(message)


if __name__ == '__main__':
    import sys
    flavor = sys.argv[1]
    telegram_key = os.getenv(f"TELEGRAM_TOKEN_{flavor.upper()}")
    if telegram_key is None:
        print(f"Please set TELEGRAM_TOKEN_{flavor.upper()}")
    bot = telebot.TeleBot(telegram_key)
    p = PersonalityBot(bot, flavor)
    def handle_messages(messages):
        for message in messages:
            # Skip non-text messages
            if message.text is None:
                continue

            try:
                p.command_dispatch(message)
            except Exception as e:
                bot_flavor = flavor
                capture_exception(e)
                print(e)
                bot.reply_to(
                    message,
                    f"⚠️ reported to sentry",
                )

    bot.set_update_listener(handle_messages)
    bot.infinity_polling()
