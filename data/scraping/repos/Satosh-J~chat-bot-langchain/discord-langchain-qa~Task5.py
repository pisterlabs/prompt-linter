import discord
from discord.ext import commands
import openai
# from langchain import LangChain
from dotenv import dotenv_values

# chain = LangChain("gpt-3.5-turbo")

intents = discord.Intents.default()
intents.typing = False
intents.presences = False

client = commands.Bot(command_prefix='!', intents=intents)

config = dotenv_values(".env")

bot_token = config["BOT_TOKEN"]
api_key = config["API_KEY"]

openai.api_key = api_key


@client.event
async def on_ready():
    print("Logged in as".format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Process the user's message
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=message.content,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )

    # Send the response back to the user
    await message.channel.send(f'Hello {response.choices[0].text}!')

    await client.process_commands(message)


def get_name_joke(name):
    # Function to generate a joke about the user's name
    return f"Why did {name} go to the store? To buy some new jokes!"


client.run(bot_token)