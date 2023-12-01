import openai
import discord
import requests
import time
import os
from PIL import Image

GUILD = '{Midjourney-Generarion-Server}'

client = discord.Client(intents = discord.Intents.default())

@client.event
async def on_ready():
    for guild in client.guilds:
        if guild.name == GUILD:
            break
    print(f'{client.user} has connected to Discord')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    elif client.user.mentioned_in(message):
        api_version = '2022-08-03-preview'
        url = "{}dalle/text-to-image?api-version={}".format(api_base, api_version)
        headers= { "api-key": api_key, "Content-Type": "application/json" }
        body = {
            "caption": message.content,
            "resolution": "1024x1024"
            }
        submission = requests.post(url, headers=headers, json=body)
        operation_location = submission.headers['Operation-Location']
        retry_after = submission.headers['Retry-after']
        status = ""
        while (status != "Succeeded"):
            time.sleep(int(retry_after))
            response = requests.get(operation_location, headers=headers)
            status = response.json()['status']
        image_url = response.json()['result']['contentUrl']
        print(image_url)

        await message.channel.send(image_url)
        print(message.content)




with open('token.txt') as f:
    # converting out text file to a list of lines
    lines = f.read().split('\n')
    # openai api key
    api_key = lines[0]
    # discord token
    discord_token = lines[1]
    api_base = lines[2]
#close the file
f.close()



client.run(discord_token)
