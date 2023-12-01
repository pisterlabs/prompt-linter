import os
from time import sleep

import discord
from dotenv import load_dotenv
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TEMPLATE = os.getenv("TEMPLATE")

template = f"{TEMPLATE}" "{history}" "Human: {human_input}" "Galen:"

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0),  # type: ignore
    prompt=prompt,
    verbose=False,
    memory=ConversationalBufferWindowMemory(k=50),
)

entry = TEMPLATE

bot = discord.Client(intents=discord.Intents.default())  # type: ignore


@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message):  # type: ignore
    if message.author == bot.user:  # type: ignore
        return

    if message.content.lower() == "hello, how are you today?":  # type: ignore
        sleep(5)
        response = chatgpt_chain.predict(human_input=entry)
        response = response.strip("\n")
    elif "bye" in message.content.lower():  # type: ignore
        response = "Goodbye, I hope you have a great day."
    else:
        sleep(5)
        response = chatgpt_chain.predict(human_input=message.content)  # type: ignore
        response = response.strip("\n")

    await message.channel.send(f"{response}")  # type: ignore


bot.run(DISCORD_TOKEN)  # type: ignore
