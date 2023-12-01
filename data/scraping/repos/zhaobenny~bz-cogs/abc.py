import re
from abc import ABC
from datetime import datetime

from openai import AsyncOpenAI
from redbot.core import Config, commands
from redbot.core.bot import Red

from aiuser.common.cache import Cache
from aiuser.messages_list.entry import MessageEntry


# for other settings to use
@commands.group(aliases=["ai_user"])
@commands.guild_only()
async def aiuser(self, _):
    """Utilize OpenAI to reply to messages and images in approved channels"""
    pass


class CompositeMetaClass(type(commands.Cog), type(ABC)):
    pass


class MixinMeta(ABC):
    def __init__(self, *args):
        self.bot: Red
        self.config: Config
        self.cached_options: dict
        self.override_prompt_start_time: dict[int, datetime]
        self.cached_messages: Cache[int, MessageEntry]
        self.ignore_regex: dict[int, re.Pattern]
        self.channels_whitelist: dict[int, list[int]]
        self.openai_client: AsyncOpenAI

    def initialize_openai_client(self):
        pass
