import discord
from discord.ext import commands
import os
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import requests
from tempfile import TemporaryDirectory
import cohere
from asgiref.sync import sync_to_async

import openai
load_dotenv()
co = cohere.Client(os.getenv('COHERE_API_KEY'))
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='%', intents=intents)
openai.api_key = os.getenv("OPENAI_API_KEY")

# chatbot = Chatbot({
#     "session_token": os.getenv('CHATGPT_ID')
# }, conversation_id=None, parent_id=None)  # You can start a custom conversation


@bot.command(name='leetcode', help='Search for a leetcode status')
async def lc(ctx, *users):
    response = ""
    users = set(user.lower() for user in users)
    for user in users:
        if (response != ""):
            response += "\n"
        # link = f'https://leetcode-stats-api.herokuapp.com/{user}'
        fetch_response = requests.get(
            f'https://leetcode-stats-api.herokuapp.com/{user}').json()
        if (fetch_response["status"] == "error"):
            response += "!! Your input for "+user + \
                " is invalid! Might because wrong user name!" + "\n"
        elif (fetch_response["status"] == "success"):
            totalSolved = fetch_response["totalSolved"]
            easySolved = fetch_response["easySolved"]
            mediumSolved = fetch_response["mediumSolved"]
            hardSolved = fetch_response["hardSolved"]
            response += f"-> User: {user}\n-- Total Solved: {totalSolved}\n-- Easy Solved: {easySolved}\n" + \
                f"-- Medium Solved: {mediumSolved}\n-- Hard Solved: {hardSolved}"
        else:
            response += "For the input "+user + \
                ", you accidentally find a cool bug! Contact Exec to report!" + "\n"
        # if not stock_exists(stock):
        #     response += "User{user}" + "\n"
        #     continue
        # response += "Price for **%s**: $%0.3f\n" % (
        #     stock, get_stock_price(stock))

    await ctx.send("```\n"+response+"```")


@bot.command(name='test', help='Search for a leetcode status')
async def test(ctx):
    await ctx.send("test success")


@bot.command(name='gpt', help='Chat with our bot about anything')
async def price(ctx, question):
    response = await sync_to_async(openai.Completion.create)(
        model="text-davinci-003",
        prompt=question,
        temperature=0.7,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    await ctx.send("```"+response["choices"][0]["text"]+"```")


@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
    # print('Ready!')


async def send_pdf_as_images(message, url):
    pdf_bytes = requests.get(url, stream=True)
    more_than_ten_pages = False
    with TemporaryDirectory() as path:
        images = convert_from_bytes(pdf_bytes.raw.read(
        ), paths_only=True, output_folder=path, grayscale=True, last_page=11, fmt='.jpg', thread_count=4)

        if len(images) > 10:
            more_than_ten_pages = True
            # NOTE: Don't return: we want to send the images anyway
            images = images[:10]
        files = [discord.File(fp=filename, filename=f"page_{idx + 1}.jpg")
                 for idx, filename in enumerate(images)]
        await message.channel.send(files=files)

    if more_than_ten_pages:
        warning_message = "**WARN**: Your PDF was more than 10 pages long: only sending the first 10 pages"
        await message.channel.send(warning_message)


async def convert_pdf(message):
    for attachment in message.attachments:
        if not attachment.filename.endswith('.pdf'):
            continue
        await send_pdf_as_images(message, attachment.url)


# @bot.event
# async def on_message(message):
#     if message.author.bot:
#         return

#     if bot.user.mentioned_in(message):
#         message_content = message.content.replace('<@1055537693112664144>', '')
#         response = await sync_to_async(openai.Completion.create)(
#             model="text-davinci-003",
#             prompt=message_content,
#             temperature=0.7,
#             max_tokens=4000,
#             top_p=1,
#             frequency_penalty=0.0,
#             presence_penalty=0.0,
#         )

#         await message.channel.send(response["choices"][0]["text"])

        # print(response)


try:
    bot.run(TOKEN)
# catch statement to print error below
except Exception as e:
    print(e)
