import asyncio
from datetime import datetime
from os import environ
from random import randint
from discord.ext import commands
import nltk
import openai
from sqlalchemy import Boolean, Column, Date, Time
from lib.prompt import Prompt
import json

from schwi.SchwiCog import SchwiCog


def forreal_factory(base):
    class ForReal(base):
        __tablename__ = "forreal"
        day = Column(Date, primary_key=True)
        time = Column(Time, primary_key=True)
        notification_sent = Column(Boolean)

        def __init__(self):
            self.notification_sent = False
            self.day = datetime.now().date()
            self.time = (
                datetime.now()
                .time()
                .replace(
                    hour=randint(0, 23), minute=randint(0, 59), second=randint(0, 59)
                )
            )

    return ForReal


class ForReal(SchwiCog):
    dependencies = ["Db"]
    models = [forreal_factory]

    @commands.Cog.listener()
    async def on_ready(self):
        self.logger.info("ForReal cog ready.")

        while await asyncio.sleep(60) or True:
            forreal = (
                self.db.Session.query(self.db.ForReal)
                .filter_by(day=datetime.now().date())
                .first()
            )

            if forreal is None:
                forreal = self.db.ForReal()
                self.db.Session.add(forreal)
                self.db.Session.commit()

            if forreal.notification_sent:
                return

            time_to_sleep = (
                datetime.combine(datetime.now().date(), forreal.time) - datetime.now()
            ).total_seconds()

            await asyncio.sleep(time_to_sleep)
            await self.send_forreal()
            forreal.notification_sent = True
            self.db.Session.commit()

    async def send_forreal(self):
        if environ.get("NODE_ENV", "development").lower() == "development":
            self.logger.info("Not sending forreal in development")
            return
        
        self.logger.info("Sending forreal")
        channel = self.schwi.get_channel(1101232564041175110)
        await channel.send("For real? @everyone")

    @commands.command(name="forreal")
    async def forreal_cmd(self, ctx):
        await self.send_forreal()
        await ctx.reply("DEBUG COCK AND BALLS")
