import time
import os
import requests
import discord
from discord.ext import commands
from marsbots_core.models import ChatMessage
from marsbots_core.resources.language_models import OpenAIGPT3LanguageModel
from marsbots_core.resources.discord_utils import (
    is_mentioned,
    remove_role_mentions,
    replace_bot_mention,
    replace_mentions_with_usernames,
    role_is_mentioned,
)
from . import settings, prompts, channels, utils


class AbrahamCog(commands.Cog):
    def __init__(self, bot: commands.bot) -> None:
        self.bot = bot
        self.language_model = OpenAIGPT3LanguageModel(
            engine=settings.GPT3_ENGINE,
            temperature=settings.GPT3_TEMPERATURE,
            frequency_penalty=settings.GPT3_FREQUENCY_PENALTY,
            presence_penalty=settings.GPT3_PRESENCE_PENALTY,
        )

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        dm = isinstance(message.channel, discord.channel.DMChannel)
        dm_allowed = dm and message.author.id in channels.ALLOWED_DM_USERS
        if (
            (
                is_mentioned(message, self.bot.user)
                or role_is_mentioned(message, channels.BOT_ROLE)
                or dm_allowed
            )
            and message.author.id != self.bot.user.id
            and (dm_allowed or message.channel.id in channels.ALLOWED_CHANNELS)
        ) and not message.author.bot:
            ctx = await self.bot.get_context(message)

            question = message.content
            url = os.environ["SERVER_URL"]
            password = os.environ["SERVER_PASSWORD"]
            folder = os.path.join(os.getcwd(), "results")
            headers = {"Content-Type": "application/json"}

            data = {"question": question, "password": password}
            result = requests.post(url + "/run", json=data, headers=headers).json()

            if "token" not in result:
                async with ctx.channel.typing():
                    await message.reply("Something went wrong :(")
                return

            token = result["token"]
            async with ctx.channel.typing():
                await message.reply("I will get back to you")

            finished = False
            while not finished:
                data = {"token": token, "password": password}
                result = requests.post(url + "/fetch", json=data, headers=headers)
                result = result.json()
                status = result["status"]["status"]
                if status == "complete":
                    output = result["output"]
                    response = output["response"]
                    filename = utils.download_file(url + "/" + output["video"], folder)
                    filepath = f"{folder}/{filename}"
                    finished = True
                    async with ctx.channel.typing():
                        local_file = discord.File(filepath, filename=filepath)
                        await message.reply(response, file=local_file)
                elif status == "failed":
                    finished = True
                    async with ctx.channel.typing():
                        await message.reply("Something went wrong :(")
                else:
                    time.sleep(1)

    def format_prompt(self, messages):
        last_message_content = replace_bot_mention(messages[-1].content).strip()
        messages_content = self.format_messages(messages)
        topic_idx = self.get_similar_topic_idx(last_message_content)
        topic_prelude = prompts.topics[topic_idx]["prelude"]
        prompt = (
            self.format_prompt_messages(prompts.prelude)
            + "\n"
            + self.format_prompt_messages(topic_prelude)
            + "\n"
            + messages_content
            + "\n"
            + "**["
            + self.bot.user.name
            + "]**:"
        )
        return prompt

    def format_prompt_messages(self, messages):
        return "\n".join(
            [
                "**[%s]**: %s" % (message["sender"], message["message"])
                for message in messages
            ]
        )

    def format_messages(self, messages_content):
        return "\n".join(
            [
                str(
                    ChatMessage(
                        self.message_preprocessor(message_content),
                        self.get_sender(message_content),
                        deliniator_left="**[",
                        deliniator_right="]**:",
                    )
                )
                for message_content in messages_content
            ]
        )

    def get_similar_topic_idx(self, query: str) -> int:
        docs = [topic["document"] for topic in prompts.topics]
        res = self.language_model.document_search(docs, query)
        return self.language_model.most_similar_doc_idx(res)

    def message_preprocessor(self, message: discord.Message) -> str:
        message_content = replace_bot_mention(message.content, only_first=True)
        message_content = replace_mentions_with_usernames(
            message_content, message.mentions
        )
        message_content = remove_role_mentions(message_content)
        message_content = message_content.strip()
        return message_content

    def get_sender(self, message: discord.Message) -> str:
        if message.author.id == self.bot.user.id:
            return self.bot.user.name
        else:
            return f"{utils.get_nick(message.author)}"


def setup(bot: commands.Bot) -> None:
    bot.add_cog(AbrahamCog(bot))
