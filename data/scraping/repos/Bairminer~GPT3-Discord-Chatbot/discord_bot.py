import asyncio
import openai
import discord
from discord.ext import commands
import traceback

from config import token, status, desc
from key import key
from chatbot import Bot

client = commands.Bot(command_prefix="BOT_NAME")
openai.api_key = key
global chatbot
chatbot = Bot(header=desc)


@client.event
async def on_ready():  # logged in to discord
    print('Logged in as ' + client.user.name + ' (ID:' + str(client.user.id) + ')')
    await client.change_presence(activity=discord.Game(name=status))


@client.listen()
async def on_message(message):  # respond to valid messages
    global chatbot
    if message.author != client.user:  # prevent bot from mentioning itself
        if not message.mention_everyone:  # ignore @everyone
            if client.user.mentioned_in(message) or isinstance(message.channel, discord.abc.PrivateChannel):  # respond if mentioned
                async with message.channel.typing():  # show that the bot is typing
                    txtinput = message.content.replace("<@" + str(client.user.id) + ">", "").replace("<@!" + str(client.user.id) + ">", "")  # remove @ mention
                    if (txtinput == " reset") or (txtinput == "reset") or (txtinput == " shut up"):  # reset bot
                        chatbot = Bot(header=desc)
                        await message.channel.send("Bot reset!")
                        return
                    try:
                        print("User: " + txtinput)
                        txt = chatbot.reply(txtinput).split("\n")
                        for line in txt:
                            if "Bot: " in line:  # remove identifier from multiline output
                                line.replace("Bot: ", "")
                            print("Bot: " + line)
                            if (line != "") and (line != "\n"):
                                await message.channel.send(line)  # send
                    except:
                        txt = traceback.format_exc()  # fetch error
                        txt = "An error has occurred. Please try again:\n```" + txt + "```"
                        await message.channel.send(txt)  # send


def main():
    loop = asyncio.get_event_loop()
    task = loop.create_task(client.start(token))
    gathered = asyncio.gather(task, loop=loop)
    loop.run_until_complete(gathered)


if __name__ == '__main__':
    main()
