import os
import discord
from dotenv import load_dotenv
import requests
import time
from utilities import (
    get_random_jack,
    is_working_hours, 
    get_random_dog, 
    save_temp_file, 
    generate_image_from_prompt,
    generate_variation,
    classify_image,
    ask_gpt,
    ask_turbo
    )
import openai
from PIL import Image


DELETE_TIME = 120

load_dotenv()
intents = discord.Intents(messages=True, guilds=True)
intents.members = True
client = discord.Client(intents=intents)
openai.api_key= os.getenv('OPEN_API_KEY')
TOKEN = os.getenv('DISCORD_TOKEN')


# TODO set flags as variables with default values so they can be changed
@client.event
async def on_ready():
    guild = client.guilds[0]
    print(f'{client.user} is connected to the following guild:\n')
    print(f'{guild.name}(id: {guild.id})')
    print(guild.channels)


@client.event
async def on_message(message):
    # Prevents bot from endlessly messaging due to reading its own messages
    if message.author == client.user:
        return
    if 'aya' in message.content:
        if not is_working_hours():
            random_dog = get_random_dog()
            save_temp_file(random_dog)
            await message.channel.send(file=discord.File(r"sample_image.png"), delete_after=DELETE_TIME)
            time.sleep(1)
            os.remove('sample_image.png')
        if is_working_hours():
            random_jack = get_random_jack()
            await message.channel.send(file=discord.File(random_jack))
    
    if message.channel.name=="gpt-3" or message.channel.name=="test":
        if '!testme' in message.content:
            try:
                image = generate_image_from_prompt(message)
                save_temp_file(image)
                await message.channel.send(file=discord.File(r"sample_image.png"))
                time.sleep(1)
                os.remove('sample_image.png')
            except:
                await message.channel.send("I cannot do that")

        if '!editme' in message.content and message.attachments:
            try:
                r = requests.get(message.attachments[0].url)
                save_temp_file(r)

                # Open user provided image, resize and save to comply with API requirements
                image = Image.open('sample_image.png')
                new_image = image.resize((1024, 1024))
                new_image.save('sample_image.png')

                await message.channel.send('Generating edited image')
                edited_image_url = generate_variation()
                edited_image = requests.get(edited_image_url)
                save_temp_file(edited_image)
                
                await message.channel.send(file=discord.File(r"sample_image.png"))
                time.sleep(1)
                os.remove('sample_image.png')
            except:
                await message.channel.send("I cannot do that")

        if '!classifyme' in message.content and message.attachments:
            r = requests.get(message.attachments[0].url)
            save_temp_file(r)
            confidence_score, classification = classify_image()
            await message.channel.send(f"I am {confidence_score}% sure that this is a {classification}")
            time.sleep(1)
            os.remove('sample_image.png')
                    
        if '!askme' in message.content:
            response = ask_gpt(message)
            await message.channel.send(response['choices'][0]['text'])

        if '!turboask' in message.content:
            response = ask_turbo(message)
            await message.channel.send(response.choices[0].message['content'])
       

client.run(TOKEN)

