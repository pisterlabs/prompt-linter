from math import e
import os
import discord
import requests
import openai
import asyncio
import re

from main import generateImage
from dallie3 import genImg
from util import random_string, downloadImage

# Load environment variables
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DISCORD_CHANNEL = os.getenv('DISCORD_CHANNEL')
GAS_URL = os.getenv('GAS_URL')

if not DISCORD_TOKEN:
    raise Exception('Discord token is missing')
if not OPENAI_API_KEY:
    raise Exception('OpenAI API key is missing')
if not DISCORD_CHANNEL:
    raise Exception('Discord channel is missing')
if not GAS_URL:
    raise Exception('GAS URL is missing')



# Initialize OpenAI
openai.api_key = OPENAI_API_KEY
sysprompt = ""

# Client configuration
intents = discord.Intents.all()
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print("Bot is ready.")

@client.event
async def on_message(message):
    if message.author.bot:
        return
    if message.channel.id != int(DISCORD_CHANNEL):
        return
    if message.content.startswith("!") or message.content.startswith("！"):
        return
    if message.content.startswith("http"):
        return

    if(message.content.startswith("_direct3")):
        print("direct dalle3", message.content.replace("_direct3", "").strip()[0:10])
        r = await genImg(message.content.replace("_direct3", "").strip(), message.content)
        if(r == ""):
            await message.reply("🍊検閲済み")
            return
        else:
            filename = random_string(10)
            save_path = await downloadImage(r, filename)
            await message.reply(file=discord.File(save_path), content=message.content.replace("_direct3", "").strip())
            return
    if(message.content.startswith("_direct")):
        await generateImage(message.content.replace("_direct", "").strip())
        await message.reply(file=discord.File("results/image.png"), content=message.content.replace("_direct", "").strip())
        return

    if message.reference:
        reply_chain = await get_reply_chain(message)
        await generate_reply_with_ref(reply_chain + [message], message)
    else:
        await generate_reply(message.content, message)

async def get_reply_chain(message):
    reply_chain = []
    current_message = message

    while current_message.reference:
        try:
            referenced_message = await message.channel.fetch_message(current_message.reference.message_id)
            reply_chain.insert(0, referenced_message)
            current_message = referenced_message
        except Exception as e:
            print('Error fetching message:', e)
            break

    return reply_chain

def get_sys_prompt():
    return sysprompt

async def get_prompt_task():
    global sysprompt
    response = requests.get(GAS_URL)
    sysprompt = response.text

async def generate_reply(user_prompt, message):
    sys = get_sys_prompt()
    response = await openai.ChatCompletion.acreate(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user_prompt.replace("-nsfw", "")}
        ]
    )
    reply = response.choices[0].message["content"]
    print(reply[:10], response.usage.total_tokens)

    pattern = r"generate\((.*?)\)"
    match = re.search(pattern, reply, re.DOTALL)

    if match:
        await generateReplyWithImage(match, user_prompt, message)
        return
    else:
        await message.reply(reply)
        return

async def generateReplyWithImage(match, user_prompt, message):
    prefix = ""
    if("-nsfw" in user_prompt or "d3" in user_prompt):
        prefix = "{{{best quality}}}, {{ultra-detailed}}, {{illustration}} {{{{masterpiece}}}}, "
    else:
        prefix = "nsfw, {{{best quality}}}, {{ultra-detailed}}, {{illustration}} {{{{masterpiece}}}}, "

    # Get the image URLs
    if ("+d3" in user_prompt):
        print(prefix + match.group(1).replace("\"",""))
        r = await genImg(prefix + match.group(1).replace("\"",""), user_prompt)
        print(r)
        if(r == ""):
            await message.reply("🍊検閲済み")
            return
        else:
            filename = random_string(10)
            save_path = await downloadImage(r, filename)
            await message.reply(file=discord.File(save_path), content=prefix + match.group(1).replace("\"",""))
            return
    await generateImage(prefix + match.group(1).replace("\"",""))
    await message.reply(file=discord.File("results/image.png"), content=prefix + match.group(1).replace("\"",""))
    return

async def generate_reply_with_ref(prompt, message):
    msgs = [
        {"role": "system", "content": sysprompt}
    ]

    for p in prompt:
        role = "assistant" if p.author.id == client.user.id else "user"
        msgs.append({
            "role": role,
            "content": p.content
        })

    response = await openai.ChatCompletion.acreate(
        model="gpt-4-1106-preview",
        messages=msgs
    )
    reply = response.choices[0].message["content"]

    pattern = r"generate\((.*?)\)"
    match = re.search(pattern, reply, re.DOTALL)

    if match:
        await generateReplyWithImage(match, prompt[-1].content, message)
        return
    else:
        print(reply[:10], response.usage.total_tokens)
        await message.reply(reply)
        return

# Set up a loop to periodically update the sysprompt
async def update_sysprompt_periodically():
    while True:
        await get_prompt_task()
        await asyncio.sleep(60)  # Wait for 60 seconds before updating again

# Start the periodic sysprompt update and the bot
async def main():
    await asyncio.gather(
        client.start(DISCORD_TOKEN),
        update_sysprompt_periodically()
    )

asyncio.run(main())
