import os
import discord
from dotenv import load_dotenv
from discord.ext import commands
import openai

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Store your OpenAI API key in an environment variable

intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents)

# Set up OpenAI API
openai.api_key = OPENAI_API_KEY


@bot.event
async def on_ready():
    guild = discord.utils.get(bot.guilds, name=GUILD)
    print(
        f'{bot.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})\n'
    )
    members = '\n - '.join([member.name for member in guild.members])
    print(f'Guild Members:\n - {members}')


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.lower() == 'hi':
        response = 'Hi'
        await message.channel.send(response)

    await bot.process_commands(message)


@bot.command(name="chat", description="Have a chat with ChatGPT")
async def chat(ctx, *, message: str):
    if bot.user == ctx.author:
        return

    username = str(ctx.author)
    current_channel = ctx.channel
    print(f"{username} : /chat [{message}] in ({current_channel})")

    try:
        # Send the user's message to ChatGPT API
        response = openai.Completion.create(
            engine="davinci",
            prompt=message,
            max_tokens=150,  # Limit the response to a certain number of tokens
            stop=["\n"],  # Stop the response at the first line break
        )

        # Get the response from the ChatGPT API
        bot_response = response.choices[0].text.strip()

        # Send the response back to the Discord channel
        await ctx.send(bot_response)

    except Exception as e:
        await ctx.send("Oops! Something went wrong while chatting with ChatGPT.")
        print(f"Error: {e}")


bot.run(TOKEN)
