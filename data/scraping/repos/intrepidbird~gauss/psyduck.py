# Has better help thing so please keep in mind

from typing import Any
import wolframalpha
import openai
import random
import asyncio

import discord
from discord import Intents
from discord.ext import commands, tasks


# Replace these with your actual API keys and bot token
openai.api_key = 'your-openai-api-key'
wolfram_api_key = 'your-wolfram-alpha-api-key'
bot_token = 'your-bot-token'

intents = Intents.all()
intents.messages = True

status = cycle(['psyduck orz', 'I am a bot', 'Ask me anything'])


class PsyduckHelpCommand(commands.HelpCommand):

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)

    async def send_bot_help(self, mapping: Any):
        embed = discord.Embed(title="Psyduck Bot Commands", description="List of available commands:", color=0x00ff00)
        embed.add_field(name="!orz <question>", value="Ask a question to Wolfram Alpha.", inline=False)
        embed.add_field(name="!ai <prompt>", value="Generate text using GPT-3 AI.", inline=False)
        embed.add_field(name="!roll <sides>", value="Roll a dice with the specified number of sides.", inline=False)

        await self.context.send(embed=embed)


bot = commands.Bot(
    command_prefix='!',
    intents=intents,
    help_command=PsyduckHelpCommand(),
)


# Error handling
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Command not found. Type `!help` to see available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Missing required argument. Please check the command usage with `!help`.")
    else:
        await ctx.send(f"An error occurred: {error}")


@bot.event
async def on_ready():
    print("[+] Psyduck is ready")


@bot.event
async def on_connect():
    change_status.start()


@tasks.loop(seconds=10)
async def change_status():
    new_status = next(status)
    await bot.change_presence(activity=discord.Game(new_status))


@bot.command(name='orz')
async def orz(ctx, *, question: str):
    try:
        client = wolframalpha.Client(wolfram_api_key)
        res = client.query(question)
        if res['@success'] == 'false':
            await ctx.send('Psyduck cannot find the answer')
        else:
            answer = next(res.results).text
            await ctx.send(f'Psyduck found out that the answer is {answer}')
    except Exception as e:
        await ctx.send('Psyduck encountered an error.')


@bot.command(name='ai')
async def gpt3(ctx, *, prompt: str):
    try:
        response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100)
        await ctx.send(response.choices[0].text.strip())
    except Exception as e:
        await ctx.send('Psyduck encountered an error.')


# Command for rolling a dice
@bot.command(name='roll')
async def roll_dice(ctx, sides: int):
    if sides <= 1:
        await ctx.send("The dice must have at least 2 sides.")
    else:
        result = random.randint(1, sides)
        await ctx.send(f"Rolled a {sides}-sided dice and got: {result}")


if __name__ == '__main__':
    # Run the bot
    bot.run(bot_token)
