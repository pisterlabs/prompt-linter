from ctypes import FormatError
import os
import openai
import random
import logging
import discord
import time
from discord.ext import commands
from asyncio import sleep
import asyncio
import colorama
from discord import app_commands
from colorama import Fore
colorama.init()


client = discord.Client(intents = discord.Intents.all())
tree = app_commands.CommandTree(client)



openai.api_key = ''
logging.basicConfig(filename="assets/log.txt", level=logging.INFO,
                    format="%(asctime)s %(message)s")


@client.event
async def on_ready():
    activity = discord.Game(name="HyperAI", type=3)
    await client.change_presence(status=discord.Status.online, activity=activity)
    await tree.sync(guild=discord.Object(id="1054463932317835285"))
    print("Ready!")




@client.event
async def on_message(message):
    if message.author == client.user:
        return
    elif message.content == '!creator':
        try: 
            await message.channel.send('HyperAI creator is Garry')
        except:
            await message.channel.send('HyperAI is under Maintaince error')
            print('An error ocured in 4-th slot')
    else:
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"{message.content}\n",
                max_tokens=500,
                temperature=1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            ).choices[0].text
            print(f'{Fore.BLUE}Author: {message.author}')
            print(f'{Fore.CYAN}Message: {message.content}')
            print(f'{Fore.GREEN}Response: {response}{Fore.RESET}')
            logging.info(f" Author = {message.author} ; Message: {message.content} ; Response: {response}")
            print('')
            await message.channel.send(response)
        except:
            await message.channel.send('HyperAI is under Maintaince error')
            print('An error ocured in 4-th slot')
    

@tree.command(name = "creator", description = "Shows who created HyperAI", guild=discord.Object(id=1054463932317835285)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def first_command(interaction):
    await interaction.response.send_message("HyperAI founder is Garry")


client.run('')

