import asyncio
from collections import deque
from collections.abc import MutableMapping
from typing import Optional

import discord
import langid
from discord.channel import VoiceChannel
from discord.ext import commands, tasks
from discord.ext.commands.context import Context
from discord.voice_client import VoiceClient

from app.error import VoiceConnectionError
from app.extension.chatgpt import OpenAIClient
from app.extension.chatgpt._client import (
    MAX_THREAD_MESSAGES,
    CompletionData,
    CompletionResult,
)
from app.services.logger import generate_log
from cogs.tts.player import TTSSource


class CustomDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.store: dict[int, VoiceClient] = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class VoiceObject(CustomDict):
    logger = generate_log()

    def __init__(self):
        super().__init__()
        self.changed = False

    def __setitem__(self, key, item):
        self.logger.info("SET %s %s", key, item)
        self.store[key] = item
        self.changed = True

    def __delitem__(self, key):
        self.logger.info("DEL %s %s", key, self.store[key])
        del self.store[key]
        self.changed = True

    def __getitem__(self, key):
        return self.store[key]

    def __repr__(self):
        return repr(self.store)

    def __len__(self):
        return len(self.store)


class TTSCore(commands.Cog):
    __slots__ = ("bot", "voice", "messageQueue")

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = generate_log()
        self.messageQueue: CustomDict[Optional[dict[int, deque]]] = CustomDict()
        self.voice = VoiceObject()
        self.lock = asyncio.Lock()
        self.volume = 150
        self.opneai = OpenAIClient()

    @staticmethod
    def func_state(r_func):
        if r_func is None:
            return False
        else:
            return r_func

    def is_joined(self, ctx: Context, member: discord.Member):
        """
        Checks if member is in a voice channel.

        Args:
            ctx: ApplicationContext
            member: ctx.author or discord.Member
        """
        if not member.voice:
            raise

        return (
            self.voice.get(ctx.author.guild.id)
            and self.voice.get(ctx.author.guild.id) is not None
            and self.voice.get(ctx.author.guild.id).channel.id
            == member.voice.channel.id
        )

    async def join(self, ctx: Context):
        """Joins a voice channel."""
        if self.is_joined(ctx, ctx.author):
            return

        # noinspection PyProtectedMember
        if self.func_state(ctx.author.guild._voice_states[ctx.author.id]):
            channel = ctx.author.guild._voice_states[ctx.author.id].channel
        else:
            await ctx.send(content="fail connect voice channel")
            raise VoiceConnectionError(
                f"Failed to join voice channel on guild {ctx.guild.id}"
            )

        if self.func_state(ctx.guild.voice_client):
            if ctx.guild.voice_client.channel.id == channel.id:
                return

            _voice = self.voice[ctx.author.guild.id]

            try:
                await _voice.move_to(channel)
            except asyncio.TimeoutError:
                await ctx.send(content=f"Moving to channel: <{str(channel)}> timed out")
                raise VoiceConnectionError(
                    f"Moving to channel: <{str(channel)}> timed out"
                )
        else:
            try:
                self.voice[ctx.author.guild.id] = await channel.connect()
            except asyncio.TimeoutError:
                await ctx.send(
                    content=f"Connecting to channel: <{str(channel)}> timed out"
                )
                raise VoiceConnectionError(
                    message=f"Connecting to channel: <{str(channel)}> timed out"
                )
            else:
                self.messageQueue[ctx.author.guild.id] = deque([])

    async def disconnect(self, ctx: Context):
        """Disconnects from voice channel."""
        if not self.voice.get(ctx.author.guild.id):
            return
        await self.voice[ctx.author.guild.id].disconnect()
        self.voice.__delitem__(ctx.author.guild.id)

    async def play(self, ctx: Context, source: discord.AudioSource):
        vc = ctx.guild.voice_client
        if not vc:
            await ctx.invoke(self.join)
        self.voice[ctx.author.guild.id].play(source)

    # async def _tts(self, ctx: Context, text: str, status):
    #    """Text to Speech"""
    #    try:
    #        if not self.voice[ctx.author.guild.id].is_playing():
    #            player = await TTSSource.text_to_speech(text)
    #            await self.play(ctx=ctx, source=player)
    #            return status(True)
    #        else:
    #            self.messageQueue[ctx.author.guild.id].append(text)
    #            while self.voice[ctx.author.guild.id].is_playing():
    #                await asyncio.sleep(1)
    #            q_text = self.message_queue[ctx.author.guild.id].popleft()
    #            q_player = await TTSSource.text_to_speech(q_text)
    #            await self.play(ctx=ctx, source=q_player)
    #            return status(True)
    #    except Exception:
    #        return status(Exception)

    async def _azure_tts(
        self,
        ctx: Context,
        text: str,
        lang: str,
        pass_text: str | None = None,
        delete_after: float | None = None,
    ):
        """Text to Speech"""
        try:
            if pass_text is not None:
                await ctx.send(
                    f"[**{ctx.author.name}**] >> {pass_text}", delete_after=delete_after
                )
            else:
                await ctx.send(
                    f"[**{ctx.author.name}**] >> {text}", delete_after=delete_after
                )

            self.messageQueue[ctx.author.guild.id].append([text, lang])

            while self.voice[ctx.author.guild.id].is_playing():
                pass
            else:
                async with self.lock:
                    self.logger.info(f"{self.messageQueue[ctx.author.guild.id]}")
                    q_text = self.messageQueue[ctx.author.guild.id].popleft()
                    q_player = await TTSSource.microsoft_azure_text_to_speech(
                        text=q_text[0], language_code=q_text[1]
                    )
                    while self.voice[ctx.author.guild.id].is_playing():
                        pass
                    else:
                        await asyncio.wait(
                            [asyncio.create_task(self.play(ctx=ctx, source=q_player))]
                        )
        except Exception as e:
            self.logger.warning(msg=f"{str(e)}")

    def discord_mention_message(self, message: discord.Message):
        if message.type == discord.MessageType.default:
            return {"role": "user", "content": message.content}
        return None

    def is_last_message_stale(self, last_message: discord.Message) -> bool:
        return (
            last_message
            and last_message.author
            and last_message.author.id != self.bot.user.id
        )

    async def _bixby(self, ctx: Context, message: str):
        channel_messages = [
            {"role": "system", "content": "Bixby"},
            {"role": "user", "content": message},
        ]
        channel_messages = [x for x in channel_messages if x is not None]
        channel_messages.reverse()
        response_data: CompletionData = self.opneai.generate_completion_response(
            message=channel_messages
        )
        status = response_data.status
        reply_text = response_data.reply_text
        status_text = response_data.status_text

        if status is CompletionResult.OK:
            u_lang = langid.classify(reply_text)[0]
            match u_lang:
                case "ko":
                    await self._azure_tts(ctx=ctx, text=reply_text, lang="ko-KR")
                case "en":
                    await self._azure_tts(ctx=ctx, text=reply_text, lang="en-US")
                case _:
                    await self._azure_tts(ctx=ctx, text=reply_text, lang="ko-KR")
        else:
            u_lang = langid.classify(status_text)[0]
            match u_lang:
                case "ko":
                    await self._azure_tts(ctx=ctx, text=reply_text, lang="ko-KR")
                case "en":
                    await self._azure_tts(ctx=ctx, text=reply_text, lang="en-US")
                case _:
                    await self._azure_tts(ctx=ctx, text=reply_text, lang="ko-KR")
