import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import openai
from discord.ext.commands.errors import ExtensionAlreadyLoaded

load_dotenv()
token = os.getenv('DISCORD_TOKEN')
#gpt_token = os.getenv('GPT_TOKEN')

intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

#openai.api_key = gpt_token
#openai.Model.list()

async def load_extensions():
    for filename in os.listdir('./core'):
        if filename.endswith('.py'):
            try:
                await bot.load_extension(f'core.{filename[:-3]}')
            except ExtensionAlreadyLoaded:
                pass

@bot.event
async def on_ready():
    print(f"Rising from the grave as {bot.user.name}")
    await load_extensions()
    fmt = await bot.tree.sync()
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.playing, name=f"with your mind"))
    #print(f"synced {len(fmt)} commands")
    print(f"Loaded: {len(bot.cogs)} core files")

@bot.event
async def on_command_error(ctx, error):
    # handle your errors here
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(f"Command not found. Use {bot.command_prefix}help to see available commands.")
    else:
        print(f'Error occurred: {error}')


if __name__ == "__main__":
    bot.run(token)