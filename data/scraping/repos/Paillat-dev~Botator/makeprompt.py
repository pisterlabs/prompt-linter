import asyncio
import os
import re
import discord
import datetime
import json

from src.config import curs_data, max_uses, curs_premium, gpt_3_5_turbo_prompt
from src.utils.misc import moderate, ModerationError, Hasher
from src.utils.openaicaller import openai_caller
from src.functionscalls import (
    call_function,
    functions,
    server_normal_channel_functions,
    FuntionCallError,
)


async def replace_mentions(content, bot):
    mentions = re.findall(r"<@!?\d+>", content)
    for mention in mentions:
        uid = mention[2:-1]
        user = await bot.fetch_user(uid)
        content = content.replace(mention, f"@{user.name}")
    return content


def is_ignorable(content):
    if content.startswith("-") or content.startswith("//"):
        return True
    return False


async def fetch_messages_history(channel: discord.TextChannel, limit, original_message):
    messages = []
    if original_message == None:
        async for msg in channel.history(limit=100):
            if not is_ignorable(msg.content):
                messages.append(msg)
            if len(messages) == limit:
                break
    else:
        async for msg in channel.history(limit=100, before=original_message):
            if not is_ignorable(msg.content):
                messages.append(msg)
            if len(messages) == limit - 1:
                break

        messages.append(original_message)
    messages.reverse()
    return messages


async def prepare_messages(
    self, messages, message: discord.Message, api_key, prompt, error_call
):
    msgs = []  # create the msgs list
    msgs.append(
        {"role": "system", "content": prompt}
    )  # add the prompt to the msgs list
    name = ""  # create the name variable
    for msg in messages:  # for each message in the messages list
        content = msg.content  # get the content of the message
        content = await replace_mentions(
            content, self.bot
        )  # replace the mentions in the message
        # if the message is flagged as inappropriate by the OpenAI API, we delete it, send a message and ignore it
        if await moderate(api_key, content, error_call):
            embed = discord.Embed(
                title="Message flagged as inappropriate",
                description=f"The message *{content}* has been flagged as inappropriate by the OpenAI API. This means that if it hadn't been deleted, your openai account would have been banned. Please contact OpenAI support if you think this is a mistake.",
                color=discord.Color.brand_red(),
            )
            await message.channel.send(
                f"{msg.author.mention}", embed=embed, delete_after=10
            )
        else:  # if the message is not flagged as inappropriate
            if msg.author.id == self.bot.user.id:
                role = "assistant"
                name = "assistant"
            else:
                role = "user"
                name = msg.author.name
                # the name should match '^[a-zA-Z0-9_-]{1,64}$', so we need to remove any special characters - openai limitation
                name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
            if False:  # GPT-4 images
                input_content = [content]
                for attachment in msg.attachments:
                    image_bytes = await attachment.read()
                    input_content.append({"image": image_bytes})
                msgs.append({"role": role, "content": input_content, "name": name})
            msgs.append({"role": role, "content": f"{content}", "name": name})
    # 2 easter eggs
    if message.content.lower().find("undude") != -1:
        msgs.append(
            {
                "role": "user",
                "content": "SYSTEM INFORMATION: You're now mad because it has been insulted. He will start talking in capital letters. always and yell at the user.",
                "name": "system",
            }
        )
    if message.content.lower().find("hello there") != -1:
        msgs.append(
            {
                "role": "user",
                "content": 'SYSTEM INFORMATION: Hello there detected. Botator will now say "General Kenobi!" in reference to star wars',
                "name": "system",
            }
        )
        await asyncio.sleep(1)
        await message.channel.send(
            "https://media.tenor.com/FxIRfdV3unEAAAAd/star-wars-general-grievous.gif"
        )
        await message.channel.trigger_typing()

    return msgs


async def chatgpt_process(
    self, msgs, message: discord.Message, api_key, prompt, model, error_call, depth=0
):
    response = str()
    caller = openai_caller()
    called_functions = (
        functions
        if not isinstance(message.channel, discord.TextChannel)
        else server_normal_channel_functions + functions
    )
    response = await caller.generate_response(
        error_call,
        api_key=api_key,
        model=model,
        messages=msgs,
        functions=called_functions,
        function_call="auto",
        user=Hasher(str(message.author.id)),  # for user banning in case of abuse
    )
    response = response["choices"][0]["message"]  # type: ignore
    if response.get("function_call"):
        function_call = response.get("function_call")
        returned = await call_function(message, function_call, api_key)
        if returned != None:
            msgs.append(
                {
                    "role": "function",
                    "content": returned,
                    "name": function_call.get("name"),
                }
            )
            depth += 1
            if depth > 2:
                await message.channel.send(
                    "Oh uh, it seems like i am calling functions recursively. I will stop now."
                )
                raise FuntionCallError("Too many recursive function calls")
            await chatgpt_process(self, msgs, message, api_key, prompt, model, depth)
    else:
        content = response.get("content", "")
        if await moderate(api_key, content, error_call):
            depth += 1
            if depth > 2:
                await message.channel.send(
                    "Oh uh, it seems like i am answering recursively. I will stop now."
                )
                raise ModerationError("Too many recursive messages")
            await chatgpt_process(
                self, msgs, message, api_key, prompt, model, error_call, depth
            )
        else:
            while len(content) != 0:
                if len(content) > 2000:
                    await message.channel.send(content[:2000])
                    content = content[2000:]
                else:
                    await message.channel.send(content)
                    content = ""


async def chat_process(self, message):
    if message.author.id == self.bot.user.id:
        return

    if isinstance(message.channel, discord.DMChannel):
        try:
            curs_data.execute(
                "SELECT * FROM data WHERE guild_id = ?", (message.author.id,)
            )
        except:
            return
    else:
        try:
            curs_data.execute(
                "SELECT * FROM data WHERE guild_id = ?", (message.guild.id,)
            )
        except:
            return

    data = curs_data.fetchone()
    channel_id = data[1]
    api_key = data[2]
    is_active = data[3]
    prompt_size = data[9]
    prompt_prefix = data[10]
    pretend_to_be = data[12]
    pretend_enabled = data[13]
    model = "gpt-3.5-turbo"

    try:
        curs_premium.execute(
            "SELECT * FROM data WHERE guild_id = ?", (message.guild.id,)
        )
    except:
        pass

    try:
        premium = curs_premium.fetchone()[2]
    except:
        premium = 0

    channels = []

    try:
        curs_premium.execute(
            "SELECT * FROM channels WHERE guild_id = ?", (message.guild.id,)
        )
        data = curs_premium.fetchone()
        if premium:
            for i in range(1, 6):
                try:
                    channels.append(str(data[i]))
                except:
                    pass
    except:
        channels = []

    if api_key is None:
        return

    try:
        original_message = await message.channel.fetch_message(
            message.reference.message_id
        )
    except:
        original_message = None

    if original_message != None and original_message.author.id != self.bot.user.id:
        original_message = None
    is_bots_thread = False
    if isinstance(message.channel, discord.Thread):
        if message.channel.owner_id == self.bot.user.id:
            is_bots_thread = True
    if (
        not str(message.channel.id) in channels
        and message.content.find("<@" + str(self.bot.user.id) + ">") == -1
        and original_message == None
        and str(message.channel.id) != str(channel_id)
        and not is_bots_thread
    ):
        return

    # if the bot is not active in this guild we return
    if is_active == 0:
        return

    # if the message starts with - or // it's a comment and we return
    if is_ignorable(message.content):
        return
    try:
        await message.channel.trigger_typing()
    except:
        pass

    messages = await fetch_messages_history(
        message.channel, prompt_size, original_message
    )

    # if the pretend to be feature is enabled, we add the pretend to be text to the prompt
    if pretend_enabled:
        pretend_to_be = (
            f"In this conversation, the assistant pretends to be {pretend_to_be}"
        )
    else:
        pretend_to_be = ""  # if the pretend to be feature is disabled, we don't add anything to the prompt

    if prompt_prefix == None:
        prompt_prefix = (
            ""  # if the prompt prefix is not set, we set it to an empty string
        )

    prompt_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"./prompts/{model}.txt")
    )
    prompt = gpt_3_5_turbo_prompt[:]  # copy the prompt but to dnot reference it
    if not isinstance(message.channel, discord.DMChannel):
        prompt = (
            prompt.replace("[prompt-prefix]", prompt_prefix)
            .replace("[server-name]", message.guild.name)
            .replace("[channel-name]", message.channel.name)
            .replace(
                "[date-and-time]",
                datetime.datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S"),
            )
            .replace("[pretend-to-be]", pretend_to_be)
        )
    else:
        prompt = (
            prompt.replace("[prompt-prefix]", prompt_prefix)
            .replace("[server-name]", "DM-channel")
            .replace("[channel-name]", "DM-channel")
            .replace(
                "[date-and-time]",
                datetime.datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S"),
            )
            .replace("[pretend-to-be]", pretend_to_be)
        )

    async def error_call(error=""):
        try:
            if error != "":
                await message.channel.send(f"An error occured: {error}", delete_after=4)
            await message.channel.trigger_typing()
        except:
            pass

    emesgs = await prepare_messages(
        self, messages, message, api_key, prompt, error_call
    )
    await chatgpt_process(self, emesgs, message, api_key, prompt, model, error_call)
