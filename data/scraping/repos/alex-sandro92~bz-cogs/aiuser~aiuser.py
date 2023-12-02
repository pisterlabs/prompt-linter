
import json
import logging
import random
import re
from datetime import datetime, timezone

import discord
import openai
from redbot.core import Config, app_commands, commands
from redbot.core.bot import Red

from aiuser.abc import CompositeMetaClass
from aiuser.common.cache import Cache
from aiuser.common.constants import (AI_HORDE_MODE, DEFAULT_PRESETS,
                                     DEFAULT_REMOVELIST, DEFAULT_REPLY_PERCENT,
                                     DEFAULT_TOPICS, MAX_MESSAGE_LENGTH,
                                     MIN_MESSAGE_LENGTH)
from aiuser.model.openai import OpenAI_LLM_Response
from aiuser.prompt_handler import PromptHandler
from aiuser.prompts.common.messageentry import MessageEntry
from aiuser.random_message_task import RandomMessageTask
from aiuser.settings.base import Settings

logger = logging.getLogger("red.bz_cogs.aiuser")


class AIUser(Settings, PromptHandler, RandomMessageTask, commands.Cog, metaclass=CompositeMetaClass):
    """ Utilize OpenAI to reply to messages and images in approved channels. """

    def __init__(self, bot):
        super().__init__()
        self.bot: Red = bot
        self.config = Config.get_conf(self, identifier=754070)
        # cached options
        self.optin_users: list[int] = []
        self.optout_users: list[int] = []
        self.optindefault: dict[int, bool] = {}
        self.channels_whitelist: dict[int, list[int]] = {}
        self.reply_percent: dict[int, float] = {}
        self.ignore_regex: dict[int, re.Pattern] = {}
        self.override_prompt_start_time: dict[int, datetime] = {}
        self.cached_messages: Cache[int, MessageEntry] = Cache(limit=100)

        default_global = {
            "custom_openai_endpoint": None,
            "optout": [],
            "optin": [],
            "ratelimit_reset": datetime(1990, 1, 1, 0, 1).strftime('%Y-%m-%d %H:%M:%S'),
            "max_topic_length": 200,
            "max_prompt_length": 200,
        }

        default_guild = {
            "optin_by_default": False,
            "reply_percent": DEFAULT_REPLY_PERCENT,
            "messages_backread": 10,
            "messages_backread_seconds": 60 * 120,
            "reply_to_mentions_replies": False,
            "scan_images": False,
            "scan_images_mode": AI_HORDE_MODE,
            "max_image_size": 2 * (1024 * 1024),
            "model": "gpt-3.5-turbo",
            "custom_text_prompt": None,
            "channels_whitelist": [],
            "public_forget": False,
            "ignore_regex": None,
            "removelist_regexes": DEFAULT_REMOVELIST,
            "parameters": None,
            "weights": None,
            "random_messages_enabled": False,
            "random_messages_percent": 0.012,
            "random_messages_topics": DEFAULT_TOPICS,
            "presets": json.dumps(DEFAULT_PRESETS)
        }
        default_member = {
            "custom_text_prompt": None,
        }
        default_channel = {
            "custom_text_prompt": None,
        }

        self.config.register_member(**default_member)
        self.config.register_guild(**default_guild)
        self.config.register_channel(**default_channel)
        self.config.register_global(**default_global)

    async def cog_load(self):
        if await self.config.custom_openai_endpoint():
            openai.api_base = await self.config.custom_openai_endpoint()
        openai.api_key = (await self.bot.get_shared_api_tokens("openai")).get("api_key")
        self.optin_users = await self.config.optin()
        self.optout_users = await self.config.optout()
        all_config = await self.config.all_guilds()

        for guild_id, config in all_config.items():
            self.optindefault[guild_id] = config["optin_by_default"]
            self.channels_whitelist[guild_id] = config["channels_whitelist"]
            self.reply_percent[guild_id] = config["reply_percent"]
            pattern = config["ignore_regex"]
            self.ignore_regex[guild_id] = re.compile(pattern) if pattern else None

        self.random_message_trigger.start()

    async def cog_unload(self):
        self.random_message_trigger.cancel()

    async def red_delete_data_for_user(self, *, requester, user_id: int):
        for guild in self.bot.guilds:
            member = guild.get_member(user_id)
            if member:
                await self.config.member(member).clear()
                # TODO: remove user messages from cache instead of clearing the whole cache
                self.cached_messages = Cache(limit=100)

    @app_commands.command(name="chat")
    @app_commands.describe(text="The prompt you want to send to the AI.")
    @app_commands.checks.cooldown(1, 30)
    @app_commands.checks.cooldown(1, 5, key=None)
    async def slash_command(self, inter: discord.Interaction, *,
                            text: app_commands.Range[str, MIN_MESSAGE_LENGTH, MAX_MESSAGE_LENGTH]):
        """ Talk directly to this bot's AI. Ask it anything you want! """
        ctx = await commands.Context.from_interaction(inter)
        ctx.message.content = text
        if not await self.is_common_valid_reply(ctx):
            return await ctx.send("You're not allowed to use this command here.", ephemeral=True)

        rate_limit_reset = datetime.strptime(await self.config.ratelimit_reset(), '%Y-%m-%d %H:%M:%S')
        if rate_limit_reset > datetime.now():
            return await ctx.send("The command is currently being ratelimited!", ephemeral=True)

        prompt_instance = await self.create_prompt_instance(ctx)
        if not prompt_instance:
            return await ctx.send("Error: Invalid message", ephemeral=True)
        prompt = await prompt_instance.get_list()
        if prompt is None:
            return await ctx.send("Error: No prompt set.", ephemeral=True)

        await OpenAI_LLM_Response(ctx, self.config, prompt).sent_response()

    @commands.Cog.listener()
    async def on_red_api_tokens_update(self, service_name, api_tokens):
        if service_name == "openai":
            openai.api_key = api_tokens.get("api_key")

    @commands.Cog.listener()
    async def on_message_without_command(self, message: discord.Message):
        ctx: commands.Context = await self.bot.get_context(message)

        if not await self.is_common_valid_reply(ctx):
            return

        if await self.is_bot_mentioned_or_replied(message):
            pass
        elif random.random() > self.reply_percent.get(message.guild.id, DEFAULT_REPLY_PERCENT):
            return

        rate_limit_reset = datetime.strptime(await self.config.ratelimit_reset(), '%Y-%m-%d %H:%M:%S')
        if rate_limit_reset > datetime.now():
            logger.debug(
                f"Want to respond but ratelimited until {rate_limit_reset.strftime('%Y-%m-%d %H:%M:%S')}")
            if await self.is_bot_mentioned_or_replied(message) or self.reply_percent.get(message.guild.id, DEFAULT_REPLY_PERCENT) == 1.0:
                await ctx.react_quietly("💤")
            return

        prompt_instance = await self.create_prompt_instance(ctx)
        if not prompt_instance:
            return
        prompt = await prompt_instance.get_list()
        if prompt is None:
            return

        await OpenAI_LLM_Response(ctx, self.config, prompt).sent_response()

    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """ Catch embeds loading """
        ctx: commands.Context = await self.bot.get_context(after)
        if not await self.is_common_valid_reply(ctx):
            return

        time_diff = datetime.now(timezone.utc) - after.created_at
        if not (time_diff.total_seconds() <= 10):
            return

        if random.random() > self.reply_percent.get(before.guild.id, DEFAULT_REPLY_PERCENT):
            return

        rate_limit_reset = datetime.strptime(await self.config.ratelimit_reset(), '%Y-%m-%d %H:%M:%S')
        if rate_limit_reset > datetime.now():
            return

        if self.contains_youtube_link(after.content):  # should be handled the first time
            return

        prompt = None
        if len(before.embeds) != len(after.embeds):
            prompt_instance = await self.create_prompt_instance(ctx)
            prompt = await prompt_instance.get_list()
        if prompt is None:
            return

        await OpenAI_LLM_Response(ctx, self.config, prompt).sent_response()

    async def is_common_valid_reply(self, ctx: commands.Context) -> bool:
        """ Run some common checks to see if a message is valid for the bot to reply to """
        if not ctx.guild:
            return False
        if await self.bot.cog_disabled_in_guild(self, ctx.guild):
            return False
        if ctx.author.bot or not self.channels_whitelist.get(ctx.guild.id, []):
            return False
        if not ctx.interaction and (
            isinstance(
                ctx.channel, discord.Thread) and ctx.channel.parent.id not in self.channels_whitelist[ctx.guild.id]
            or ctx.channel.id not in self.channels_whitelist[ctx.guild.id]
        ):
            return False
        if not await self.bot.ignored_channel_or_guild(ctx):
            return False
        if not await self.bot.allowed_by_whitelist_blacklist(ctx.author):
            return False
        if ctx.author.id in self.optout_users:
            return False
        if not self.optindefault.get(ctx.guild.id) and ctx.author.id not in self.optin_users:
            return False
        if self.ignore_regex.get(ctx.guild.id) and self.ignore_regex[ctx.guild.id].search(ctx.message.content):
            return False

        if not openai.api_key:
            await self.initalize_openai(ctx)
        if not openai.api_key:
            return False

        return True

    async def is_bot_mentioned_or_replied(self, message: discord.Message) -> bool:
        if not (await self.config.guild(message.guild).reply_to_mentions_replies()):
            return False
        if self.bot.user in message.mentions:
            return True
        if message.reference and message.reference.message_id:
            reference_message = message.reference.cached_message or await message.channel.fetch_message(message.reference.message_id)
            return reference_message.author == self.bot.user
        return False

    async def initalize_openai(self, ctx: commands.Context):
        openai.api_key = (await self.bot.get_shared_api_tokens("openai")).get("api_key")
        if not openai.api_key:
            await ctx.send(
                f"OpenAI API key not set for `aiuser`. "
                f"Please set it with `{ctx.clean_prefix}set api openai api_key,API_KEY`")
