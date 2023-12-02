import json
import random
import re
from pathlib import Path

import openai
from discord import Message, File
from discord.ext import commands
from discord.ext.commands import Cog

async def setup(bot):
    tts = Tts(bot)


class Tts(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        print("TTS cog init")

    # on message
    @Cog.listener("on_message")
    async def on_message(self, message):
        if message.author.bot:
            return

        if message.content.startswith("!*"):
            self.bot.logger.info("TTS command detected")
            file = await self.generate_ai_audio(message.content[1:])
            await message.channel.send(file=file)


    async def cog_load(self):
        print("GPT Chat cog loaded")

    async def generate_ai_audio(self, input_text):
        voice = random.choice(["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
        print("Generating audio with voice: " + voice)
        speech_file_path = Path(__file__).parent / f"audio/speech-{voice}.mp3"
        openai.api_key = self.bot.config['OPENAI_API_KEY']

        response = openai.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=input_text
        )

        response.stream_to_file(speech_file_path)

        with open(speech_file_path, "wb") as f:
            f.write(response.content)

        # To return as a Discord file object
        discord_file = File(speech_file_path)
        return discord_file

