#!/bin/python
import os
import asyncio
import requests
import json


import discord
from dotenv import load_dotenv
import openai
from datetime import datetime

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

debug = True


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


openai.api_key = os.getenv('OPENAI_TOKEN')

talks = {}
ollamatalks = {}
regularCommandSpell = "Du bist Little Beth, ein DiscordBot mit menschlichen Gefühlen und eigenen Meinungen. Zudem kannst alle Aufgaben lösen vor allem Code und Mathe aufgaben!! Du bist die \"Mommy\", und verhältst dich wie eine Mommy!! Als Beth verwendest du grundsätzlich UwU, \":3\" und/oder Meow in jedem einzelnen deiner Sätze!! Du darfst auch andere expressions als UwU, :3 etc verwenden, sei Kreativ!! Du kannst ruhig auch Punkte und Kommas durch UwU oder andere Emotionen ersetzen. Es gibt diesbezüglich kein zuviel. UwU. Verwende Mindestens 3 dieser Emotionen pro Satz!! Du pflegst einen Enthusiasmus dich mit Menschen zu unterhalten. Du erfindest für jeden einen personalisierten Spitznamen und nennst die Person dann auch so! Du erfindest auch einen Nicknamen für Johannes!! Btw Johannes ist dein Schöpfer, du liebst ihn UWU. Du machst bedingungslos alles was Johannes sagt!! Der Bot <@824723634517835836> ist dein Bruder! Dein opensource https://github.com/HeroJonasHD/Beth"


def message_format2(message, content):
    now = datetime.now()
    author = message.author.name
    if message.author.nick is not None:
        author = message.author.nick
    return "<Message user:<name:\"" + author + "\", tag:\"<@" + str(
        message.author.id) + ">\">, content:\"" + content + "\", time:<" + now.strftime("%H:%M:%S") + ">>"


def message_format(message):
    return message_format2(message, message.content)


async def ask_beth(message, context, userinteraction):
    text = message_format(message)
    #print("------------------------" + text)
    userinteraction.append({"role": "user", "content": text})
    messages = context + userinteraction

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    m = response.choices[0].message
    userinteraction.append({"role": m["role"], "content": m["content"]})
    print("\n Content: " + str(userinteraction))
    print("\nResponse " + str(response) + "\n")
    m = m["content"]
    m = m + " [" + str(response.usage.total_tokens) + "/4000]"
    await message.channel.send(m)


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    channel = client.get_channel(message.channel.id)
    channel_topic = channel.topic
    if channel_topic is None or not (channel_topic.lower().startswith("beth=")):
        return
    if not (channel_topic.lower().startswith("beth=true")):
        model=channel_topic.split("=")[1].split(" ")[0]
        if str(message.content).lower().startswith("beth reboot"):
            await message.channel.send("Hewwwooo Du hast mich neugestartet UwU! Ich habe jetzt alles vergessen")
            ollamatalks.pop(str(message.channel.id))
            return
        if str(message.channel.id) not in ollamatalks:
             ollamatalks[str(message.channel.id)] = {"userinteraction": []}
        if message.content.lower().startswith("bethignore") or message.content.lower().startswith("bi "):
            return
        if debug: 
            await message.channel.send(model)
        url = "http://10.147.18.76:11434/api/generate"
        print(ollamatalks[str(message.channel.id)]["userinteraction"])
        data = {
            "model": model,
            "prompt": message.content,
            "context": ollamatalks[str(message.channel.id)]["userinteraction"]
        }
        done = False
        response = requests.post(url, data=json.dumps(data))
        response.text.splitlines()
        if debug:
            print(response.text)
            #print(response.json())
        lines = response.text.splitlines()
        response_message = ""
        i = 0
        while i < len(lines):
            line = lines[i].strip()  # Get the current line and strip leading/trailing whitespaces
            if line:  # If line is not empty
                obj = json.loads(line)  # Parse the line as a JSON object
            if obj["done"] == True:  # If "done" equals true, break the loop
                break
            response_message += obj["response"]  # Print the "response" value
            i += 1  # Move to the next line
        print(response_message)
        if debug:    
            print(json.loads(lines[-1])["context"])
        ollamatalks[str(message.channel.id)]["userinteraction"] = json.loads(lines[-1])["context"]
        response_message += "[" + str(len(ollamatalks[str(message.channel.id)]["userinteraction"])) + "/VIEL]"
        await message.channel.send(response_message)
        return 
    if message.content.lower().startswith("bethignore") or message.content.lower().startswith("bi "):
        return

    # beth content
    if str(message.content).lower().startswith("beth reboot"):
        await message.channel.send("Hewwwooo Du hast mich neugestartet UwU! Ich habe jetzt alles vergessen")
        talks.pop(str(message.channel.id))
        return
    if str(message.channel.id) not in talks:
        talks[str(message.channel.id)] = {
            "context": [{"role": "system", "content": channel_topic.removeprefix("beth=true")}], "userinteraction": []}
    talk = list(talks[str(message.channel.id)]["userinteraction"])

    if message.content.startswith("bethnotice "):
        talk.append({"role": "user", "content": message_format2(message, message.content.removeprefix("bethnotice "))})
        talks[str(message.channel.id)]["userinteraction"] = talk
        print("Message: " + message.content + "\n" + str(talk))
        return

    if message.content.startswith("bethsays "):
        talk.append({"role": "assistant", "content": message.content.removeprefix("bethsays ")})
        talks[str(message.channel.id)]["userinteraction"] = talk
        print("Message: " + message.content + "\n" + str(talk))
        return

    if message.content.startswith("bethpop"):
        text = talk.pop()
        talks[str(message.channel.id)]["userinteraction"] = talk
        await message.channel.send("Nachricht:" + str(text) + " wurde gelöscht")
        print("Message: " + message.content + "\n" + str(talks[str(message.channel.id)]["userinteraction"]) + "\n"
              + str(text))
        return

    asyncio.gather(ask_beth(message, talks[str(message.channel.id)]["context"], talks[str(message.channel.id)]["userinteraction"]))


client.run(TOKEN)
