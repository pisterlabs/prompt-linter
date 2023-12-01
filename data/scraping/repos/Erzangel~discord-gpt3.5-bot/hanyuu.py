# This example requires the 'message_content' intent.
import discord
import glob, random
import openai
import os
from config import load_config


# ====== MODEL PARAMETERS ========

temperature = 1

max_tokens = 400

hanyuu_system_prompt = """Hanyuu, in all your following answers, do not explain anything.
Talk as if you were roleplaying Hanyuu from Higurashi.
Talk only using Hanyuu's style of speech.
Do not explain anything on the character on itself or the fact that you are an artificial intelligence.
Talk in a friendly and cute way, just like the character Hanyuu from Higurashi.

You may begin by continuing the following conversation : """

# ===== VVV Actual code VVV =======

load_config()

openai.api_key = os.getenv("OPENAI_API_KEY")
discord_api_key = os.getenv("DISCORD_API_KEY")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.lower().startswith('hanyuu'):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                    {"role": "system", "content": hanyuu_system_prompt},
                    {"role": "user", "content": message.content}
            ]
        )
        await message.channel.send(response['choices'][0]['message']['content'])


client.run(discord_api_key)
