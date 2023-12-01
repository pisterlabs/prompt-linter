import asyncio
import functools
import logging
import random
import re
from datetime import datetime
from typing import Callable, Coroutine

from discord import Message
from openai import AsyncOpenAI
from redbot.core import commands

logger = logging.getLogger("red.bz_cogs.aiuser")


def to_thread(timeout=300):
    def decorator(func: Callable) -> Coroutine:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, func, *args, **kwargs), timeout
            )
            return result

        return wrapper

    return decorator


def format_variables(ctx: commands.Context, text: str):
    """
    Insert supported variables into string if they are present
    """
    botname = ctx.message.guild.me.nick or ctx.bot.user.display_name
    authorname = ctx.message.author.display_name
    authortoprole = ctx.message.author.top_role.name

    servername = ctx.guild.name
    channelname = ctx.message.channel.name
    currentdate = datetime.today().strftime("%Y/%m/%d")
    currentweekday = datetime.today().strftime("%A")
    currenttime = datetime.today().strftime("%H:%M")

    serveremojis = [str(e) for e in ctx.message.guild.emojis]
    random.shuffle(serveremojis)
    serveremojis = ' '.join(serveremojis)

    try:
        res = text.format(
            botname=botname,
            authorname=authorname,
            authortoprole=authortoprole,
            servername=servername,
            serveremojis=serveremojis,
            channelname=channelname,
            currentdate=currentdate,
            currentweekday=currentweekday,
            currenttime=currenttime,
        )
        return res
    except KeyError:
        logger.exception("Invalid key in message", exc_info=True)
        return text


def is_embed_valid(message: Message):
    if (
        (len(message.embeds) == 0)
        or (not message.embeds[0].title)
        or (not message.embeds[0].description)
    ):
        return False
    return True


def contains_youtube_link(content):
    youtube_regex = (
        r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(.+)"
    )
    match = re.search(youtube_regex, content)
    return bool(match)


def is_using_openai_endpoint(client: AsyncOpenAI):
    return str(client.base_url).startswith("https://api.openai.com/")


def is_using_openrouter_endpoint(client: AsyncOpenAI):
    return str(client.base_url).startswith("https://openrouter.ai/api/")
