# Create a bot object for Discord
import nextcord
import openai
from nextcord.ext import tasks, commands
import os
import json

intents = nextcord.Intents.all()
intents.guild_messages = True

bot = commands.Bot(command_prefix='/', intents=nextcord.Intents.all())
messages_in_channel = []
messages_in_dm = []
conversations = {}

# Create a client object for OpenAI
openai.api_key = '' 

async def chat(messages):
    try:
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Your answers must be clear and concise."
                },
                *messages
            ]
        )
        text = chat_completion["choices"][0]["message"]["content"]
        return text

    except Exception as e:
        print(e)
        return "Something went wrong."

async def save_conversation(user_id, messages):
    with open(f"{user_id}_conversation.json", "w") as file:
        json.dump(messages, file, indent=4)

async def load_conversation(user_id):
    try:
        with open(f"{user_id}_conversation.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

async def save_server_list(servers):
    with open("server_list.json", "w") as file:
        json.dump(servers, file, indent=4)

async def load_server_list():
    try:
        with open("server_list.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

@bot.event
async def on_ready():
    print(f'{bot.user} is ready!')
    activity = nextcord.Activity(type=nextcord.ActivityType.playing, name="Slide into my DMs and prepare to be amazed.")
    await bot.change_presence(activity=activity)
    servers = await load_server_list()
    servers.extend([guild.id for guild in bot.guilds])
    await save_server_list(list(set(servers)))

@bot.event
async def on_guild_join(guild):
    servers = await load_server_list()
    servers.append(guild.id)
    await save_server_list(list(set(servers)))

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if message.channel.id == 1080650292657405972:
        user_messages = await load_conversation(message.author.id)
        user_messages.append({"role": "user", "content": f"{message.author.top_role}: {message.content}"})
        await save_conversation(message.author.id, user_messages)
        response = await chat(user_messages)
        await message.reply(response)

    elif message.channel.type == nextcord.ChannelType.private:
        user_messages = await load_conversation(message.author.id)
        user_messages.append({"role": "user", "content": f"{message.author}: {message.content}"})
        await save_conversation(message.author.id, user_messages)
        response = await chat(user_messages)
        await message.author.send(response)

    else:
        await bot.process_commands(message)

# Run the bot with your own bot token 
bot.run('')
