import time
import asyncio
import random
from threading import Lock

import discord
import openai
from discord.ext.commands import AutoShardedBot, CommandNotFound
from loguru import logger
from opencc import OpenCC

from .utils import get_chatgpt_config


class FriesBot(AutoShardedBot):
    def __init__(self, **kwargs):
        from fries import (
            CrystalBallMeow,
            Dice,
            EasyCalculator,
            FortuneMeow,
            FriesSummoner,
            MeowTalk,
            ResponseTemplate,
            SixtyJiazi,
            TarotMeow,
            WikiMan,
        )

        self.dice = Dice
        self.resp_template = ResponseTemplate()
        self.meow_talk = MeowTalk()
        self.fries_summoner = FriesSummoner()
        self.fortune_meow = FortuneMeow()
        self.tarot_meow = TarotMeow()
        self.calculator = EasyCalculator()
        self.wiki = WikiMan()
        self.sixty_jiazi = SixtyJiazi()
        self.crystal = CrystalBallMeow()

        chatgpt_config = get_chatgpt_config()
        openai.api_key = chatgpt_config["api_token"]
        openai.organization = chatgpt_config["organization"]
        self.cc_conv = OpenCC("s2t")
        self.delim = chatgpt_config["delim"]

        activity = discord.Activity(
            name="/薯條喵喵喵",
            type=discord.ActivityType.playing,
        )

        AutoShardedBot.__init__(
            self,
            help_command=None,
            activity=activity,
            **kwargs,
        )

    def is_need_break(self, msg: str):
        for d in self.delim:
            if msg.endswith(d):
                return True

        return False

    def get_gpt_response(self, prompt: str):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt.strip()}],
            stream=True,
        )
        ts = time.perf_counter() - 1
        msg = ""
        words = list()
        for word in response:
            try:
                words.append(word["choices"][0]["delta"]["content"])
                msg = "".join(words)
                msg = self._preprocess_msg(msg)
                delta = time.perf_counter() - ts
                if self.is_need_break(msg) and delta > 1:
                    yield msg
                    ts = time.perf_counter()
            except:
                pass

        yield msg + "\n\n喜歡這則解牌的話，請幫本喵按個 😘"

    def _preprocess_msg(self, msg: str) -> str:
        msg = msg.strip("「」")
        msg = msg.replace("。喵喵", "，喵喵")
        msg = msg.replace("纔能", "才能")
        return msg.strip()

    async def on_message(self, msg: discord.Message):
        if msg.author == self.user:
            return

        if msg.content.startswith("！"):
            msg.content = "!" + msg.content[1:]

        if msg.content.startswith("!"):
            log_type = "msglog"
            if msg.guild is None:
                log_type = "msglog2"
            logger.info(self.resp(log_type).format(msg))
            if "薯條" in msg.content:
                await msg.channel.send("現在改為斜線指令囉！請輸入 /薯條喵喵喵 獲得更多資訊")
        elif self.user in msg.mentions or msg.guild is None:
            logger.info(self.resp("msglog").format(msg))
            await self.chatting(msg)

        await AutoShardedBot.on_message(self, msg)

    async def on_ready(self):
        logger.info(f"{self.user} | Ready")

    async def on_command_error(self, _, error):
        if isinstance(error, CommandNotFound):
            return
        logger.info(str(error).replace("\n", " | "))

    def resp(self, key, *args):
        return self.resp_template.get_resp(key, *args)

    def get_pictures(self, n):
        return self.fries_summoner.get_pictures(n)

    def get_fortune(self):
        return self.fortune_meow.get_fortune()

    def get_tarots(self, n):
        return self.tarot_meow.get_tarots(n)

    def get_gpt_tarots(self, problem):
        return self.tarot_meow.get_gpt_prompt(problem)

    def query_card(self, query):
        return self.tarot_meow.query_card(query)

    def do_calc(self, formula):
        return self.calculator.calc(formula)

    def get_wiki(self, *args):
        return self.wiki.get_response(*args)

    def get_sixty_jiazi(self):
        return self.sixty_jiazi.pick()

    def get_crystal(self):
        return self.crystal.get()

    def roll_dice(self, dice, name):
        return self.dice.roller(dice, name)

    async def chatting(self, msg):
        async with msg.channel.typing():
            try:
                emojis = "🤔😂😊🤣😍😘😁😉😎"
                await msg.add_reaction(random.choice(emojis))
            except:
                pass
            await asyncio.sleep(0.5)
        await msg.channel.send(self.meow_talk.get_sent() + "\n現在改為斜線指令囉！")
