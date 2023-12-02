import asyncio
import json
import os
import sys

import discord
from discord.ext.commands import Cog, Bot

import openai
from openai.error import RateLimitError

from utils import *
from typing import *
from pathlib import Path


class botEvents(Cog):
    def __init__(self, bot: Bot):
        self.bot = bot

    @Cog.listener()
    async def on_member_join(self, member: discord.Member):
        await member.send(f"Hi {member.name}, welcome to {member.guild.name}!")

    @Cog.listener()
    async def on_member_remove(self, member: discord.Member):
        await member.send(f"Hi {member.name}, see ya :3 you have been kick!")

    @Cog.listener()
    async def on_member_ban(self, guild: discord.Guild, user: discord.Member):
        await user.send(
            f"Hi {user.name}, see ya :3 you have been ban from {guild.name}!"
        )

    @Cog.listener()
    async def on_member_unban(self, guild: discord.Guild, user: discord.Member):
        await user.send(
            f"Hi {user.name}, Your ban in {guild.name} have been revoke, you can join back!"
        )

    @Cog.listener()
    async def on_member_update(self, before: discord.Member, after: discord.Member):
        print("Some thing just happend")

    @Cog.listener()
    async def on_ready(self):
        print(f"Logged in as {self.bot.user} (ID: {self.bot.user.id})")
        print("------")
