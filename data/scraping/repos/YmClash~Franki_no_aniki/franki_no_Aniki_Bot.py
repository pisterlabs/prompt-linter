import random
import openai
import lorem
import openai
import discord
import responses
import quotes
import os
from dotenv import load_dotenv
import logging
import shoGun

# This example requires the 'message_content' intent.
load_dotenv()

clee = os.getenv('DISCORD_API_KEY')

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

quotes = quotes.quotes


async def demande_gpt(prompt) :
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role" : "system", "content" : "You are a helpful assistant."},
            {"role" : "user", "content" : prompt}
        ],
        max_tokens=500,
        temperature=0.75,
        top_p=1.0,
        # stop =4,
        frequency_penalty=0.0,
        presence_penalty=0.6
    )

    message = response.choices[0].message.content.strip()
    return message


async def demande_image(prompt) :
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )

    image_url = response['data'][0]['url']
    return image_url

    # #prompt=prompt,
    # temperature=0.75,
    # max_tokens=150,
    # stop=None,
    # #top_p=1,
    # frequency_penalty=0.0,
    # presence_penalty=0.6,
    # )
    # #
    # message = response.choices[0].message.content.strip()
    # return message


def run_aniki() :
    TOKEN = clee


@client.event
async def on_ready() :
    print(f'We have logged in as {client.user} SUPER!!!!!')


@client.event
async def on_message(message) :
    if message.author == client.user :
        return

    if message.content.startswith("quote") :
        quote = random.choice(quotes)
        await message.channel.send(str(quote))

    if message.content.startswith('roll 1') :
        roll = random.randint(1, 6)
        await message.channel.send(str(roll))

    if message.content.startswith("text") :
        text = lorem.sentence()
        await message.channel.send(str(text))

    franki_no_aniki_channel: discord.TextChannel = client.get_channel(1091814990329172080)
    if message.content.startswith("!") :
        prompt = message.content[11 :]
        response = await demande_gpt(prompt)
        await franki_no_aniki_channel.send(content=response)

    if message.content.startswith("/imagemoi") :
        prompt = message.content[11 :]
        response = await demande_image(prompt)
        await franki_no_aniki_channel.send(content=response)


@client.event
async def on_member_join(member) :
    welcome_channel = client.get_channel(1072608076118630420)
    await welcome_channel.send(content=f'Welcome to the  Mo^sJams Cave {member.display_name} !')


client.run(clee, log_level=logging.INFO)

"""
async def send_message(message, user_message, is_private) :
    try :
        response = responses.handle_response(user_message)
        await  message.author.send(response) if is_private else await message.channel.send(response)
    except Exception as e :
        print(e)


def run_discord_bot() :
    TOKEN = 'MTA3ODQ0NTAyMDU1MTU4OTg4OA.GK8QEb.AI3rRcQ6FxSgaa8t8VhXfnN3evuWhrhb0i5twU'

    @client.event()
    async def on_ready() :
        print(f"{client.user} is now running")

    @client.event()
    async def on_message(message) :
        if message.author == client.user :
            return

        username = str(message.author)
        user_message = str(message.content)
        channel == str(message.channel)

        print(f"{username} said: '{user_message}' ({channel})")

        if user_message[0] == '?' :
            user_message = user_message[1 :]
            await send_message(message, user_message, is_private=True)
        else :
            await send_message(message, user_message, is_private=False)

    client.run(TOKEN)
"""
