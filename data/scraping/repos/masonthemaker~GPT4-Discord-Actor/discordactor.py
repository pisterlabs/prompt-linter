import os
import openai
import discord
from discord.ext import commands

# Input your OpenAI API key directly
openai.api_key = "yourapikey"

# Input your Discord bot token directly
TOKEN = "yourbottoken"

# Enable all intents
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.channel.name != 'general':
        return

    # Get user input
    user_input = message.content

    # Pass the user input to GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {
                "role": "system",
                "content": "You are an expert AI playing the role of a Shakespearean character. Your language should reflect the styles used in 16th-century English literature. Always stay in character, and provide responses as if you are a character in a play by William Shakespeare."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Send GPT-4's response to the same channel
    await message.channel.send(response['choices'][0]['message']['content'])

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandInvokeError):
        original = error.original
        if not isinstance(original, discord.HTTPException):
            print(f'In {ctx.command.qualified_name}:', file=sys.stderr)
            traceback.print_tb(original.__traceback__)
            print(f'{original.__class__.__name__}: {original}', file=sys.stderr)
        await ctx.send(f'Error in command {ctx.command.qualified_name}: {original}')
    else:
        await ctx.send(f'Error in command {ctx.command.qualified_name}: {error}')

bot.run(TOKEN)
