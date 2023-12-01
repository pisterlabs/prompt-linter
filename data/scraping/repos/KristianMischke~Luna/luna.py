import json
import random
from typing import Optional

import discord
import os
import datetime
import openai
from transformers import pipeline, set_seed

from helper_functions import get_user_handle_for_gpt3
from luna_commands import LunaCommands


class Luna:

    def __init__(self):
        #
        # CONSTS
        #
        self.MAX_SECONDS_ALLOWED_REPLY = 45
        self.CONVERSATION_REPLY_PROB = 0.7
        self.MAX_CONTEXT_MESSAGES = 20

        #
        # setup discord
        #
        intents: discord.Intents = discord.Intents.default()
        intents.members = True
        intents.guilds = True
        intents.reactions = True

        self.client = discord.Client(intents=intents)
        self.commands = LunaCommands(self)

        self.gpt2_generator = pipeline('text-generation', model='gpt2')
        set_seed(42)

        #
        # other setup
        #
        with open("/Users/kristianmischke/Documents/LUNA_DATA/users.json", "rb") as f:
            self.user_data = json.load(f)
        self.user_data = {int(k): v for k, v in self.user_data.items()}
        print(self.user_data)

        self.last_message_sent_in_channel = {}

        self.r = random.Random()
        self.mode = "gpt3"
        self.commands.setup_models()
        openai.api_key = os.getenv("OPENAI_API_KEY")


luna = Luna()


@luna.client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(luna.client))


@luna.client.event
async def on_message(message: discord.Message):
    if message.author == luna.client.user:
        return

    if message.content.lower().startswith("!stop"):
        luna.last_message_sent_in_channel[message.channel.id] = None
        return

    # try to run a command
    success = await luna.commands.try_command(message)
    if success:
        return

    # general response?
    await try_response(message)


async def try_response(message: discord.Message) -> None:
    if not await should_reply(message):
        return

    global luna

    channel: discord.TextChannel = message.channel

    message_context = await get_context_messages(message)
    users = get_users_in_context(message_context)
    max_context = get_max_context_allowed(users)
    max_model = get_max_gpt3_access(users)

    actual_message_context = message_context[-max_context:]  # get the most recent messages from the context

    if luna.mode == "gpt2":
        prepared_context = prepare_context_for_gpt3(actual_message_context, users)
        print(prepared_context)
        print()

        results = luna.gpt2_generator(prepared_context, max_length=len(prepared_context)+100, num_return_sequences=1)
        response = results[0]["generated_text"][len(prepared_context):]
        response = response.split("\n")[0].strip()
        print(response)
        print()
        await channel.send(content=response)

    elif luna.mode == "gpt3":
        prepared_context = prepare_context_for_gpt3(actual_message_context, users)

        print(prepared_context)
        print()
        print(f"{max_model}\ncontext: {max_context}")

        if max_model is not None:
            response = openai.Completion.create(
                engine=max_model,
                prompt=prepared_context,
                temperature=0.9,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.6,
                stop=["\n", " Kristian:", " Luna:"]
            )

            print(response)

            await channel.send(content=response["choices"][0]["text"])

    luna.last_message_sent_in_channel[message.channel.id] = datetime.datetime.now()


async def should_reply(message: discord.Message) -> bool:

    # we are mentioned, start conversation
    if any(luna.client.user.id == m.id for m in message.mentions):
        return True

    if "luna" in message.content.lower():
        return True

    # context for specific channel
    if message.channel.id in luna.last_message_sent_in_channel \
            and luna.last_message_sent_in_channel[message.channel.id] is not None:
        current_time = datetime.datetime.now()
        time_diff = current_time - luna.last_message_sent_in_channel[message.channel.id]

        # we are in a conversation in this channel
        if time_diff.seconds < luna.MAX_SECONDS_ALLOWED_REPLY:

            # question mark in this or prev message
            if message.content[-1] == '?':
                return True
            async for prev_message in message.channel.history(limit=1):
                if prev_message.content[-1] == '?':
                    return True

            # random to respond
            return luna.r.random() < luna.CONVERSATION_REPLY_PROB

    return False


async def get_context_messages(message: discord.Message) -> list[discord.Message]:
    channel: discord.TextChannel = message.channel
    msg_list = []
    async for message in channel.history(limit=luna.MAX_CONTEXT_MESSAGES):
        msg_list.insert(0, message)
    return msg_list


def get_users_in_context(messages: list[discord.Message]) -> list[discord.Member]:
    return list(set([m.author for m in messages]))


def get_max_context_allowed(users: list[discord.Member]) -> int:
    max_context = 0
    for user in users:
        if user.id in luna.user_data:
            allowed_context = luna.user_data[user.id]["gpt3_max_context_msgs"]
            if allowed_context > max_context:
                max_context = allowed_context
    return max_context


def get_max_gpt3_access(users: list[discord.Member]) -> Optional[str]:
    max_model = None
    for user in users:
        if user.id in luna.user_data:
            davinci_allowed = luna.user_data[user.id]["gpt3_davinci_access"]
            if davinci_allowed:
                return "davinci"

            curie_allowed = luna.user_data[user.id]["gpt3_curie_access"]
            if curie_allowed and max_model is None:
                max_model = "curie"

    return max_model


def prepare_context_for_gpt3(message_context: list[discord.Message], users) -> str:
    # TODO: insert Luna mood sentiment here
    if luna.r.random() < 0.9:
        result = "Luna is a very helpful AI created by Kristian. Luna enjoys helping her friends."
    else:
        result = "Luna is a sentient AI who occasionally loves to tease her friends and sometimes be sarcastic, otherwise she is very helpful."

    for user in users:
        if user.id in luna.user_data and "luna_lore" in luna.user_data[user.id]:
            if luna.r.random() < 0.3:
                result += " " + luna.user_data[user.id]["luna_lore"]

    result += "\n"

    # TODO: insert summarized context here

    for m in message_context:
        short_handle = get_user_handle_for_gpt3(luna, m.author)

        # TODO: custom cleaning for @ mentions and emoji

        result += f"{short_handle}: {m.clean_content}\n"

    result += "Luna:"
    return result


# def replace_user_mentions_for_gpt3(message: discord.Message) -> str:
#     result = message.content
#     for user in message.mentions:
#         result = result.replace(user.mention, get_user_handle_for_gpt3(user))
#     return result


# def replace_emoji_for_gpt3(message: discord.Message) -> str:
#     message.clean
#     pass


luna.client.run(os.environ["DISCORD_KBOT_TOKEN"])
