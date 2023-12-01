import datetime
import discord
from discord.ext import commands
from marsbots_core.models import ChatMessage
from marsbots_core.programs.lm import complete_text
from marsbots_core.resources.language_models import OpenAIGPT3LanguageModel
from marsbots_core.resources.discord_utils import (
    get_reply_chain,
    get_discord_messages,
    is_mentioned,
    replace_bot_mention,
    remove_role_mentions,
    replace_mentions_with_usernames,
    role_is_mentioned,
)
from . import prompts, channels, settings


def get_nick(obj):
    if hasattr(obj, "nick") and obj.nick is not None:
        return obj.nick
    else:
        return obj.name


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
        try:
            dm = isinstance(message.channel, discord.channel.DMChannel)
            dm_allowed = dm and message.author.id in channels.DM_ALLOWED_USERS
            if (
                (
                    is_mentioned(message, self.bot.user)
                    or role_is_mentioned(message, channels.BOT_ROLE)
                )
                and message.author.id != self.bot.user.id
                and (dm_allowed or message.channel.id in channels.ALLOWED_CHANNELS)
            ) and not message.author.bot:
                ctx = await self.bot.get_context(message)
                async with ctx.channel.typing():
                    prompt = await self.format_prompt(ctx, message)
                    completion = await complete_text(
                        self.language_model, prompt, max_tokens=80, stop=["<", "\n\n"],
                        use_content_filter=True
                    )
                    await message.reply(completion.strip())

        except Exception as e:
            print(f"Error: {e}")
            await message.reply(":)")

    async def format_prompt(
        self, 
        ctx: commands.context, 
        message: discord.Message
    ) -> str:
        last_message_content = self.message_preprocessor(message)
        topic_idx = self.get_similar_topic_idx(last_message_content)
        topic_prefix = prompts.topics[topic_idx]["prefix"]
        #print(f" -> last message: {last_message_content}")
        #print(f' -> search result: {prompts.topics[topic_idx]["document"]}')
        last_messages = await get_discord_messages(ctx.channel, 1)
        reply_chain = await get_reply_chain(ctx, message, depth=6)
        if reply_chain:
            reply_chain = self.format_reply_chain(reply_chain)
        last_message_text = str(
            ChatMessage(
                f"{self.message_preprocessor(last_messages[0])}",
                "M",
                deliniator_left="<",
                deliniator_right=">",
            )
        ).strip()
        prompt = topic_prefix
        if reply_chain:
            prompt += f"{reply_chain}\n"
        prompt += "\n".join(
            [
                last_message_text,
                "<Abraham>",
            ]
        )
        return prompt

    def format_reply_chain(self, messages):
        reply_chain = []
        for message in messages:
            if message.author.id == self.bot.user.id:
                sender_name = "Chatsubo"
            else:
                sender_name = "M"
            reply_chain.append(
                ChatMessage(
                    content=f"{self.message_preprocessor(message)}",
                    sender=sender_name,
                    deliniator_left="<",
                    deliniator_right=">",
                )
            )
        return "\n".join([str(message).strip() for message in reply_chain])

    def get_similar_topic_idx(self, query: str) -> int:
        docs = [topic["document"] for topic in prompts.topics]
        res = self.language_model.document_search(docs, query)
        return self.language_model.most_similar_doc_idx(res)

    def message_preprocessor(self, message: discord.Message) -> str:
        message_content = replace_bot_mention(message.content, only_first=True)
        message_content = replace_mentions_with_usernames(
            message_content, message.mentions
        )
        message_content = message_content.strip()
        return message_content

    def get_sender(self, message: discord.Message) -> str:
        if message.author.id == self.bot.user.id:
            return self.bot.user.name
        else:
            return f"{get_nick(message.author)}"


def setup(bot: commands.Bot) -> None:
    bot.add_cog(AbrahamCog(bot))
