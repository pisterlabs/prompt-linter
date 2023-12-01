import discord
from discord.ext import commands
import random
import openai
import os

TOKEN = os.getenv("TOKEN")
APIKEY = os.getenv("APIKEY")

intents = discord.Intents.default()
intents.typing = False
intents.presences = False

bot = commands.Bot(command_prefix="/", intents=intents)
openai.api_key = APIKEY

# Read quotes from the text file
with open("quotes.txt", "r") as file:
    quotes = file.read().splitlines()

def generate_rick_quote():
    quote = random.choice(quotes)
    while not quote:
        quote = random.choice(quotes)
    return quote

@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="Snake Jazz"))
    print(f"Logged in as {bot.user.name} ({bot.user.id})")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)

@bot.tree.command(
    name="rickquote",
    description="Generates a random Rick quote."
)
async def rick(interaction: discord.Interaction):
    await interaction.response.send_message(generate_rick_quote())

@bot.tree.command(
    name="whatwouldricksay",
    description="Responds to the last message in the channel as Rick Sanchez."
)
async def rick_respond(interaction: discord.Interaction):
    channel = interaction.channel
    # Initialize a list to store the messages
    messages = []
    # Use an async for loop to get the last 2 messages
    async for message in channel.history(limit=2):
        messages.append(message)
    # Reverse the list because the messages are retrieved in reverse order
    messages.reverse()
    # Get the second-to-last message (the one before the command)
    last_message = messages[1].content if len(messages) > 1 else "I have nothing to say."
    # Add a period to the end of the message if it doesn't end with a punctuation mark
    if last_message[-1] not in {'.', '!', '?'}:
        last_message += '.'
    # Add the instructions to the prompt
    prompt = f"You are now Rick Sanchez. You talk exactly using his tone and mannerisms. Respond to the text after this sentence as Rick Sanchez. {last_message}"
    # Use GPT3 to generate a response
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=1,
        max_tokens=600
    )
    # Replace "morty" with the interaction user's name in the response
    response_text = response['choices'][0]['text'].strip().replace("Morty", interaction.user.name)
    await interaction.response.send_message(response_text)
    
@bot.tree.command(
    name="rick",
    description="Asks Rick Sanchez for his opinion."
)
async def rick_opinion(interaction: discord.Interaction, topic: str):
    # Add the instructions to the prompt
    prompt = f"You are now Rick Sanchez. You talk exactly using his tone and mannerisms. Respond to the text after this sentence as Rick Sanchez. {topic}:"
    # Use GPT-3 to generate a response
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=1,
        max_tokens=600
    )
    # Replace "morty" with the interaction user's name in the response
    response_text = response['choices'][0]['text'].strip().replace("Morty", interaction.user.name)
    await interaction.response.send_message(response_text)
    
@bot.tree.command(
    name="drunkrick",
    description="Asks a drunk Rick Sanchez for his opinion."
)
async def drunk_rick(interaction: discord.Interaction, topic: str):
    # Add the instructions to the prompt
    prompt = f"You are now a drunk, angry Rick Sanchez. You talk exactly using his tone and mannerisms. Respond to the text after this sentence as drunk angry Rick Sanchez. {topic}:"
    # Use GPT-3 to generate a response
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=1,
        max_tokens=600
    )
    # Replace "morty" with the interaction user's name in the response
    response_text = response['choices'][0]['text'].strip().replace("Morty", interaction.user.name)
    await interaction.response.send_message(response_text)

bot.run(TOKEN)
