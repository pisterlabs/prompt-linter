import aiohttp
import disnake
from disnake.ext import commands
from kani.engines.openai import OpenAIClient
from openai import AsyncOpenAI

from . import config


class Calypso(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openai_kani = OpenAIClient(api_key=config.OPENAI_API_KEY, http=aiohttp.ClientSession(loop=self.loop))
        self.openai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.enc_chatterboxes = dict()

    async def close(self):
        await self.openai_kani.close()
        await self.openai.close()
        await super().close()

    async def get_or_fetch_channel(self, channel_id: int):
        """Get a channel or thread by ID, fetching it from the API if not in cache (e.g. archived thread)."""
        channel = self.get_channel(channel_id)
        if channel is not None:
            return channel
        try:
            channel = await self.fetch_channel(channel_id)
        except disnake.NotFound:
            return None
        return channel
