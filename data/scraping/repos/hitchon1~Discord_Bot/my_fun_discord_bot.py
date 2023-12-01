
import openai
from discord import Intents
import discord

# Replace with your own Discord API token
TOKEN = "insert your discord token"
# Replace with your own OpenAI API key
openai.api_key = "insert OPENAI token"

# Create an intents object with the required intents
intents = Intents.default()
intents.members = True
# Initialize the Discord client
client = discord.Client(intents=intents)
# Initialize the Discord client


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    message_id = message.id
    channel_id = message.channel.id
    chatbot = openai.Completion()
    channel = client.get_channel(channel_id)
    channel.fetch_message(message_id)
    channel = client.get_channel(channel_id)
    message = await channel.fetch_message(message_id)
    human_message = message.content
    print(human_message)

    ai_response = chatbot.create(engine="text-davinci-003", prompt = human_message, max_tokens=2220)
    await message.channel.send(ai_response.choices[0].text)

client.run(TOKEN)
