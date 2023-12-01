import discord
import json
import openai
import os
import requests
from io import BytesIO
from discord.ext import commands

# Loading token and key
with open("config.json") as json_file:
    data = json.load(json_file)
tkn = data["token"]
openai.api_key = data["key"]

# Intent Declaration
client = discord.Client(intents=discord.Intents.all())


def process_command(message):
    '''
    Processes user commands that begin with "!"

    Args: 
        message.content (str): full command

    Returns:
        signal (list): signal[0] = operation, signal[1] = operand
            Codes:
                signal[0] == 0: 
                    operation: Send signal[1] to message channel
                    operand: Message

                signal[0] == 1: 
                    operation: Generate image based on prompt
                    operand: Prompt
    '''
    command = message.content.lstrip("!")
    signal = [0] # signal[0] signals how to operate return and operand (signal[1])

    if command in ("help", "help "):
        embed = discord.Embed(title="Commands",
        color=0xeee657)
        commands = [("!gen-p [prompt]",
            "Generates an image based on prompt input."),
            ("!gen-u", "Generates a special image just for your mom.")]
        for name, value in commands:
            embed.add_field(name=name, value=value, inline=False)
        signal.append(embed)


    elif command == "gen-u": # generate random art image - returns a unique prompt
        signal[0] = 1
        signal.append(openai.Completion.create(model="text-davinci-003", prompt="Generate a short, extremely unique and creative image \
            generation prompt", temperature=0.7, max_tokens=100).choices[0].text)

    elif command.startswith("gen-p"): # generate img based on prompt - returns user prompt
        signal[0] = 1
        signal.append(command.lstrip("gen-p ")) # signal[1] = user prompt
      
    else: # command not recognized
        signal.append("Sorry, I couldn't recognize your command. Please check your message or enter \"!help\".")
    
    return signal


def get_image(prompt):
    '''
    Prompts GPT for an image

    Args:
        prompt (str): image generation prompt

    Returns:
        image (discord.File): the image file
    '''
    
    response = openai.Image.create(
        prompt=prompt,
        size="1024x1024", 
        response_format="url",
        )
    
    image_url = response['data'][0]["url"]

    return image_url


@client.event
async def on_ready():
    await client.change_presence(status=discord.Status.online)
    print("AFD LIVE")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith("!"): # command listener
        signal = process_command(message)

        if signal[0] == 0: # if help
            if type(signal[1]) == str:
                await message.channel.send(signal[1])
            else:
                await message.channel.send(embed=signal[1])
            
            
            
        else:
            image_url = get_image(signal[1])

            embed = discord.Embed(title=signal[1],
            color=discord.Colour(0x03b1fc))
            embed.set_image(url=image_url)

            await message.channel.send(embed=embed)

client.run(tkn)
