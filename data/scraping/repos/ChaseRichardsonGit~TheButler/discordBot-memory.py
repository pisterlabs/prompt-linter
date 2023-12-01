import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAIChat
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


load_dotenv()
DISCORD_BOT_TOKEN = os.environ.get("Jarvis_TOKEN")
BOT_NAME = 'Jarvis'

intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

os.environ["OPENAI_API_KEY"] = "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"
os.environ["GOOGLE_CSE_ID"] = "a7cd60a9b57f34133"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAY2V7r-EOshEw0RMjI4ZGArxrEycnxiNk"

prefix_messages = [{"role": "system", "content": "You are a helpful discord Chatbot."}]

llm = OpenAIChat(model_name='gpt-3.5-turbo', 
             temperature=0.5, 
             prefix_messages=prefix_messages,
             max_tokens = 2000)
tools = load_tools(["google-search", "llm-math"], llm=llm)
agent = initialize_agent(tools,
                         llm,
                         agent="zero-shot-react-description",
                         verbose=True)


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if BOT_NAME.lower() in message.content.lower():
        print("Detected bot name in message:", message.content)
        
        # Capture the output of agent.run() in the response variable
        response = agent.run(message.content) 

        messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="Hi AI, how are you today?"),
                    AIMessage(content="I'm great thank you. How can I help you?"),
                    HumanMessage(content="I'd like to understand string theory.")
]



        while response:
            chunk, response = response[:2000], response[2000:]
            await message.channel.send(chunk)
        print("Response sent.")
    else:
        print("Bot name not detected in message:", message.content)

    await bot.process_commands(message)


if __name__ == "__main__":
    print("Starting the bot...")
    bot.run(DISCORD_BOT_TOKEN)