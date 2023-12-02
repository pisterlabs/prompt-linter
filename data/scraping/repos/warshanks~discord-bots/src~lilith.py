import datetime
import discord
import openai
from discord.ext import commands

from config import Lilith_TOKEN, command_sync, openai_token, openai_org
from cogs.chat_cog import LilithCog
from cogs.music_cog import MusicCog

# Get the start time of the program
start_time = datetime.datetime.now()

# Set OpenAI API key and organization
openai.api_key = openai_token
openai.organization = openai_org

# Initialize a new Discord bot instance
bot = commands.Bot(command_prefix="~", intents=discord.Intents.all())


# Event handler for when the bot is ready to use
@bot.event
async def on_ready():
    try:
        await bot.add_cog(MusicCog(bot))
        await bot.add_cog(LilithCog(bot))
        await command_sync(bot)
    except Exception as e:
        print(e)
    # Calculate the time it took for the client to start
    print("Ready in:", datetime.datetime.now() - start_time)


# Run the bot
bot.run(Lilith_TOKEN)
