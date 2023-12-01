# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:17:30 2023

@author: shangfr
"""
import json
import logging
import random
import discord
from langchain.llms import OpenAI
from vecdb import load_vectordb

with open('.streamlit/config.json', 'r') as f:
    config = json.load(f)
    
handler = logging.FileHandler(
    filename='discord.log', encoding='utf-8', mode='w')

TOKEN = config['discord']
llm = OpenAI(openai_api_key=config['openai'],
             model_name="gpt-3.5-turbo", temperature=0)

vectordb = load_vectordb(directory='fables_db')
retriever = vectordb.as_retriever(search_type="mmr")
results = retriever.get_relevant_documents("猫和老鼠")

test = [result.page_content for result in results]


class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        if message.author == client.user:
            return
        print(f'Message from {message.author}: {message.content}')

        if message.content == '!测试':
            response = random.choice(test)

            await message.channel.send(response)

        if '寓言' in message.content:
            question = message.content
            response = retriever.get_relevant_documents(question)[
                0].page_content
            output = response.split('\n')

            #### Create the initial embed object ####
            embedVar = discord.Embed(
                title=output[4], url="https://aesopica.readthedocs.io/en/latest/", description=output[5], color=0x109319)

            # Add author, thumbnail, fields, and footer to the embed

            # embedVar.set_author(name=message.author.name)
            embedVar.set_image(url="http://oss-cdn.shangfr.site/fables.png")

            # embedVar.set_thumbnail(url="https://img2.baidu.com/it/u=2024274349,3703499800&fm=253&fmt=auto&app=138&f=JPEG?w=100&h=100")
            #embedVar.add_field(name="Field 1 Title", value="This is the value for field 1. This is NOT an inline field.", inline=False)
            #embedVar.add_field(name="Field 2 Title", value="It is inline with Field 3", inline=True)
            #embedVar.add_field(name="Field 3 Title", value="It is inline with Field 2", inline=True)
            #file = discord.File("parchment.png", filename="output.png")
            embedVar.set_footer(
                text=output[6], icon_url="http://oss-cdn.shangfr.site/parchment.png")

            await message.channel.send(embed=embedVar)

        if message.content.startswith('/fable'):
            question = message.content.replace("/fable", "")
            response = retriever.get_relevant_documents(question)[
                0].page_content
            output = response.split('\n')

            embedVar = discord.Embed(
                title=output[1], url="https://aesopica.readthedocs.io/en/latest/", description=output[2], color=0x109319)
            embedVar.set_image(url="http://oss-cdn.shangfr.site/fables.png")
            embedVar.set_footer(
                text=output[3], icon_url="http://oss-cdn.shangfr.site/parchment.png")

            await message.channel.send(embed=embedVar)

        if message.content.startswith('/chat'):
            question = message.content.replace("/chat", "")
            response = llm(question)

            await message.channel.send(response)


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(TOKEN, log_handler=handler, log_level=logging.DEBUG)
