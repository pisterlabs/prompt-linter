import fileinput
import string
import nextcord as discord
import os
import sqlite3 as sql
import datetime
import random
import aiohttp
import io
import aiofiles
import openai
import asyncio
import contextlib


from nextcord.ext import commands
from nextcord.ext.commands import cooldown, BucketType
from nextcord import SlashOption, Interaction
from nextcord.ext import application_checks
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

github_token = os.getenv("GITHUB_API_TOKEN")  
token = os.getenv("BOT_TOKEN")
openai_key = os.getenv("OPENAI_API_KEY")




class Errors(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    


    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.reply("You are missing the required permissions to run this command")
        elif isinstance(error, commands.BotMissingPermissions):
            await ctx.reply("I am missing the required permissions to run this command")
        elif isinstance(error, commands.NotOwner):
            await ctx.reply("You are not the owner of this bot!")
        elif isinstance(error, commands.CommandOnCooldown):
            await ctx.reply("Cool down. (hah, get it?)\n Please try again in `{}` seconds".format(round(error.retry_after, 1)))
        elif isinstance(error, commands.DisabledCommand):
            await ctx.reply("This command is disabled.")
        elif isinstance(error, commands.NoPrivateMessage):
            await ctx.reply("This command cannot be used in private messages.")
        elif isinstance(error, commands.CheckFailure):
            await ctx.reply("You do not have the permissions to use this command.")
        elif isinstance(error, commands.CommandInvokeError):
            await ctx.reply("An error occured while running this command")
        else:
            await ctx.reply("An error occured while running this command")





def setup(bot):
    bot.add_cog(Errors(bot))
    print("Error handling cog loaded")