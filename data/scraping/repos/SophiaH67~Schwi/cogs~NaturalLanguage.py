from discord.ext import commands
import nltk
import openai
from lib.prompt import Prompt
import json

from schwi.SchwiCog import SchwiCog


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("words")
nltk.download("wordnet")
nltk.download("stopwords")


class NaturalLanguage(SchwiCog):
    answer_on = [
        "compute",
        "what",
        "who",
        "why",
        "when",
        "where",
        "how",
        "can",
        "should",
        "write",
        "which",
        "do i",
        "explain",
        "does",
        "is that",
        "is it",
        "is this",
        "schwi",
        "##compute##",
    ]

    dependencies = ["Db", "Context", "Settings"]

    @property
    async def engine(self):
        return await self.settings.get_or_create_setting(
            "natural_language_engine", "text-davinci-002"
        )

    @property
    async def temperature(self):
        return await self.settings.get_or_create_setting(
            "natural_language_temperature", "0.8"
        )

    @property
    async def max_length(self):
        return await self.settings.get_or_create_setting(
            "natural_language_max_length", "64"
        )

    @property
    async def presence_penalty(self):
        return await self.settings.get_or_create_setting(
            "natural_language_presence_penalty", "1.5"
        )

    @property
    async def frequency_penalty(self):
        return await self.settings.get_or_create_setting(
            "natural_language_frequency_penalty", "0.8"
        )

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.schwi.user:
            self.logger.debug("Not answering to myself.")
            return

        # Force answer
        if message.content == "##compute##":
            await self.answer_message(message)
            return

        # Mention check
        for mention in message.mentions:
            if mention == self.schwi.user:
                await self.answer_message(message)
                return

        # Static trigger words
        for answer_on in self.answer_on:
            if message.content.lower().startswith(answer_on):
                await self.answer_message(message)
                return
        
        # Check is "schwi" is in the message
        if "schwi" in message.content.lower():
            await self.answer_message(message)
            return

    async def answer_message(self, message):
        context = self.context.get_context(message.channel.id)
        prompt = Prompt(self.schwi)
        await prompt.add_introduction()
        await prompt.add_chat_context(message, context)

        # So that GPT doesn't generate stuff for other people
        recent_authors = list(map(lambda message: message.author.name, context))
        recent_unique_authors = list(set(recent_authors))

        recent_unique_authors = list(recent_unique_authors)
        stop_tokens = []
        for recent_unique_author in recent_unique_authors[::-1]:
            stop_tokens.append(f"{recent_unique_author}:")
            if len(stop_tokens) > 2:
                break
        self.logger.debug(f"Stop tokens: {stop_tokens}")

        response = (
            openai.Completion.create(
                engine=await self.engine,
                prompt=str(prompt),
                temperature=float(await self.temperature),
                max_tokens=int(await self.max_length),
                stop=stop_tokens,
                presence_penalty=float(await self.presence_penalty),
                frequency_penalty=float(await self.frequency_penalty),
            )
            .choices[0]
            .text
        )
        self.logger.debug(f"Prompt: {json.dumps(str(prompt))}")
        self.logger.debug(f"Response: {response}")
        response = response.replace("\n", " ").strip()
        await message.reply(response)
