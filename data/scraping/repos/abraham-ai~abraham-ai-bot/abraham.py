import asyncio
import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

import discord
from discord.ext import commands
from marsbots.discord_utils import get_discord_messages
from marsbots.discord_utils import get_reply_chain
from marsbots.discord_utils import is_mentioned
from marsbots.discord_utils import replace_bot_mention
from marsbots.discord_utils import replace_mentions_with_usernames
from marsbots.language_models import complete_text
from marsbots.language_models import OpenAIGPT3LanguageModel
from marsbots.models import ChatMessage
from marsbots.util import hex_to_rgb_float
from marsbots_eden.eden import get_file_update
from marsbots_eden.eden import poll_creation_queue
from marsbots_eden.eden import request_creation

from . import config
from . import prompts
from . import settings

CONFIG = config.config_dict[config.stage]
ALLOWED_GUILDS = CONFIG["guilds"]
ALLOWED_CHANNELS = CONFIG["allowed_channels"]
ALLOWED_RANDOM_REPLY_CHANNELS = CONFIG["allowed_random_reply_channels"]
ALLOWED_DM_USERS = CONFIG["allowed_dm_users"]


class Abraham(commands.Cog):
    def __init__(self, bot: commands.bot) -> None:
        self.bot = bot
        self.language_model = OpenAIGPT3LanguageModel(
            engine=settings.GPT3_ENGINE,
            temperature=settings.GPT3_TEMPERATURE,
            frequency_penalty=settings.GPT3_FREQUENCY_PENALTY,
            presence_penalty=settings.GPT3_PRESENCE_PENALTY,
        )

    @commands.slash_command(guild_ids=ALLOWED_GUILDS)
    async def complete(
        self,
        ctx,
        prompt: discord.Option(str, description="Prompt", required=True),
        max_tokens: discord.Option(
            int,
            description="Maximum number of tokens to generate",
            required=False,
            default=100,
        ),
    ):
        if not self.perm_check(ctx):
            await ctx.respond("This command is not available in this channel.")
            return
        await ctx.defer()
        try:
            completion = await complete_text(self.language_model, prompt, max_tokens)
            formatted = f"**{prompt}**{completion}"
            await ctx.respond(formatted)
        except Exception as e:
            logging.error(e)
            await ctx.respond("Error completing text - " + str(e))

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        try:
            if (
                message.channel.id not in ALLOWED_CHANNELS
                or message.author.id == self.bot.user.id
                or message.author.bot
            ):
                return

            trigger_reply = is_mentioned(message, self.bot.user) or (
                message.channel.id in ALLOWED_RANDOM_REPLY_CHANNELS
                and random.random() < settings.RANDOM_REPLY_PROBABILITY
            )

            if trigger_reply:
                ctx = await self.bot.get_context(message)
                async with ctx.channel.typing():
                    #chat = await self.get_chat_messages(ctx, message)
                    prompt = await self.format_prompt(ctx, message)
                    completion = await complete_text(
                        self.language_model,
                        prompt, #chat
                        max_tokens=200,
                        stop=["\n\n", "\n"],
                        use_content_filter=True,
                    )
                    await message.reply(completion.strip())

        except Exception as e:
            print(f"Error: {e}")
            await message.reply(":)")

    def message_preprocessor(self, message: discord.Message) -> str:
        message_content = replace_bot_mention(message.content, only_first=True)
        message_content = replace_mentions_with_usernames(message_content, message.mentions)
        message_content = message_content.strip()
        return message_content

    async def get_chat_messages(
        self,
        ctx: commands.context,
        message: discord.Message,
    ) -> str:
        last_messages = await get_discord_messages(ctx.channel, 1)
        reply_chain = await get_reply_chain(ctx, message, depth=8)
        chat = [{
            "role": "system", 
            "content": settings.MANIFEST
        }]
        for reply in reply_chain + last_messages:
            if reply.author.id == self.bot.user.id:
                sender_name = "assistant" # reply.author.name
            else:
                sender_name = "user"
            chat.append({
                'role': sender_name,
                'content': self.message_preprocessor(reply) 
            })
        return chat

    async def format_prompt(
        self,
        ctx: commands.context,
        message: discord.Message,
    ) -> str:
        last_message_content = self.message_preprocessor(message)
        topic_idx = 0 #self.get_similar_topic_idx(last_message_content)
        topic_prefix = prompts.topics[topic_idx]["prefix"]
        last_messages = await get_discord_messages(ctx.channel, 1)
        reply_chain = await get_reply_chain(ctx, message, depth=8)
        if reply_chain:
            reply_chain = self.format_reply_chain(reply_chain)
        last_message_text = str(
            ChatMessage(
                f"{self.message_preprocessor(last_messages[0])}",
                "M",
                deliniator_left="<",
                deliniator_right=">",
            ),
        ).strip()
        prompt = topic_prefix
        if reply_chain:
            prompt += f"{reply_chain}\n"
        prompt += "\n".join(
            [
                last_message_text,
                "<Abraham>",
            ],
        )
        return prompt

    def format_reply_chain(self, messages):
        reply_chain = []
        for message in messages:
            if message.author.id == self.bot.user.id:
                sender_name = "Abraham"
            else:
                sender_name = "M"
            reply_chain.append(
                ChatMessage(
                    content=f"{self.message_preprocessor(message)}",
                    sender=sender_name,
                    deliniator_left="<",
                    deliniator_right=">",
                ),
            )
        return "\n".join([str(message).strip() for message in reply_chain])

    def get_similar_topic_idx(self, query: str) -> int:
        docs = [topic["document"] for topic in prompts.topics]
        res = self.language_model.document_search(query, docs)
        return self.language_model.most_similar_doc_idx(res)

    async def get_start_gen_message(self, ctx):
        async with ctx.channel.typing():
            completion = await complete_text(
                self.language_model,
                prompts.start_prompt,
                max_tokens=100,
                stop=["\n", "\n\n"],
            )
            return completion

    def perm_check(self, ctx):
        if ctx.channel.id not in ALLOWED_CHANNELS:
            return False
        return True


def setup(bot: commands.Bot) -> None:
    bot.add_cog(Abraham(bot))
