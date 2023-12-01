#!/usr/bin/env python3
import discord
import openai
import re
from keys import discord_key
from keys import openai_api_key

intents = discord.Intents.all()
client = discord.Client(command_prefix="!", intents=intents)


@client.event
async def on_ready():
    print("We have logged in as {0.user}".format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("/p "):
        try:
            openai.api_key = openai_api_key
            model = "text-davinci-003"
            max_token_value = 3000
            prompt_value = message.content[3:]
            re.sub("<.*?>", "", prompt_value)
            print(prompt_value)

            # create a completion
            completion = openai.Completion.create(
                engine=model, prompt=prompt_value, max_tokens=max_token_value
            )

            # print the completion
            await message.channel.send(completion.choices[0].text)
        except Exception as e:
            print(e)
            await message.channel.send(e)


client.run(discord_key)
