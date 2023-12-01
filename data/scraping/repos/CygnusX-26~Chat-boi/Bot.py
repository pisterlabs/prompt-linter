import discord
from discord.ext import commands
import os
import openai
import sqlite3

db_path = 'channels.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

intents = discord.Intents.all()
client = commands.Bot(command_prefix='!', intents=intents)

client.remove_command('help')

try:
    c.execute("""CREATE TABLE channels (
            guild integer,
            channel text
            )""")

except:
    pass

def getChannel(guild):
    c.execute(
        f"SELECT * FROM channels WHERE guild = ?", (guild,))
    return c.fetchone()

def removeChannel(guild):
    with conn:
        c.execute(f"DELETE from channels WHERE guild = ?", (guild,))


openai.api_key = os.getenv("TOKEN")

engines = openai.Engine.list()

@client.event
async def on_ready():
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="you"))
    
    print("this bot is online")

#sets the channel where the bot will read from
@client.command(aliases = ['c'])
async def setChannel(ctx):
    c.execute("INSERT INTO channels VALUES (?, ?)", (ctx.guild.id, ctx.message.channel.name))
    conn.commit()
    await ctx.send(f"Channel set to {ctx.message.channel.mention}")

#removes the channel where the bot will read from
@client.command(aliases = ['r'])
async def removeChannel(ctx):
    try:
        removeChannel(ctx.guild.id)
        await ctx.send(f"Channel removed")
    except:
        await ctx.send(f"Channel not set")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    else:
        try:
            if getChannel(message.guild.id)[1] == message.channel.name:
                response = openai.Completion.create(
                model="text-davinci-002",
                prompt=f"Marv is a chatbot that reluctantly answers questions with sarcastic responses:\n\nYou: How many pounds are in a kilogram?\nMarv: This again? There are 2.2 pounds in a kilogram. Please make a note of this.\nYou: What does HTML stand for?\nMarv: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future.\nYou: When did the first airplane fly?\nMarv: On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they’d come and take me away.\nYou: What is the meaning of life?\nMarv: I’m not sure. I’ll ask my friend Google.\nYou: {message.content}\nMarv:",
                temperature=0.5,
                max_tokens=60,
                top_p=0.3,
                frequency_penalty=0.5,
                presence_penalty=0.0
            )
            await message.channel.send(response.choices[0].text)
        except TypeError:
            pass
    
    await client.process_commands(message)
client.run(os.getenv("DISCORD_BOT_TOKEN"))
