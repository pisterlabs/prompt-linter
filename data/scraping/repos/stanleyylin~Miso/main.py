import discord
from discord.ext import commands
from commands import Commands
import misogynyModel
import cohere
import asyncio

intents = discord.Intents.all()
# client = commands.Bot()

client = discord.Client(intents= intents)

client = commands.Bot(command_prefix = '+', intents= intents)

# TODO: learn about cogs 

@client.event 
async def on_ready():
    print("The bot is now ready for use.")
    print("-----------------------------")

@client.event
async def on_message(message):
    #if message.author.bot:
    #    return
    await asyncio.sleep(0) 
    curFile = open("messageHistory.txt","a")
    curFile.write(str(message.id) + ",")
    curFile.write(str(message.content) + ",")
    curFile.write(str(message.author.name) + "\n")

    if message.author.bot:
        curFile.write(str(message.id) + ",")
        curFile.write(str(message.content) + ",")
        curFile.write(str(message.author.name) + "\n")
        return    
    
    problemList = misogynyModel.classifyMessage(message.content)

    #problemList = [true/false, >60, toxic/benign, >90]
    print(problemList)
    if str(problemList[0]) == "True" and int(problemList[1]) >= 60:
        await message.delete()
        await message.channel.send("Please take a moment to learn why that last message was misogynistic: https://stopsexism.carrd.co/")
    elif str(problemList[2]) == "Toxic" and int(problemList[3] >=80):
        await message.delete()
        await message.channel.send("Please take a moment to learn why that last message was toxic: https://stoptoxicity.carrd.co/")
    else:
        await asyncio.sleep(3) 
    
    import requests
    url = "https://api.estuary.tech/content/add"



    import os
    fileName = "messageHistory" + str(message.id) + ".txt"
    curFile = open(fileName,"w")
    payload={}
    files=[
    ('data',('file',open('/Users/pranav/git/repository/Miso/' + fileName + '','rb'),'application/octet-stream'))
    ]
    headers = {
    'Accept': 'application/json',
    'Authorization' : "Bearer EST0bcf3718-9ad8-44b5-b893-eee73f497ce6ARY"
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    # print(response.json()['cid'])
    #initialled used this ^ line to save the content IDs, but it works too inconsistently when requesting from the server
    # to keep it in the code. as a short term fix we saved another file for message history as well
    curFile.close()
    try:
        os.remove(fileName)
    except:
        print("file not found")
    

@client.command()
async def hello(ctx):
    print("received message")
    await ctx.send("hola")

with open("token.txt", "r", encoding="utf-8") as f:
    bottoken1 = f.read()

client.run(bottoken1)