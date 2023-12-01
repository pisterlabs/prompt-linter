import os
import discord
from dotenv import load_dotenv
import openai
import asyncio
import daemonocle
import logging

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

from discord.ext import commands
#from discord_slash import SlashCommand, SlashContext

# get the grand-parent directory of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')

# load the environment variables from the .env file
load_dotenv(dotenv_path=env_path)

TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.default()
intents.message_content = True
#client = discord.Client(intents=intents)
client = commands.Bot(command_prefix='!', intents=intents)
#slash = SlashCommand(client, sync_commands=True)

CHANNEL_ID=1060974735346901056 #PoweredOn#main
channel_histories = {};

openai.api_key = os.getenv("OPENAI_API_KEY")

import signal
import sys

print("hello "+__file__ + ":" + __name__)

def count_self_tokens():
    filename = os.path.abspath(__file__)
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
        token_count = len(enc.encode(content))
    print(f"Number of OpenAI API tokens in {filename}: {token_count}")
    return token_count


# Define a shutdown callback function
async def shutdown_callback():
    logging.info("\nSetting bot status to offline...")
    # Set the bot's status to offline before shutdown
    await client.change_presence(status=discord.Status.offline)
    await client.close()
    logging.info("Bot is offline.")
    # Add your cleanup code here

def get_combo_id(message) -> str:
    channel_id = message.channel.id
    thread_id = message.reference.message_id if message.reference else ''
    combo_id = f"{channel_id}-{thread_id}"
    return combo_id

# Add your main code here
#while True:
    #logging.debug("Running...")
    # Your code here

async def get_thread_or_channel_history(message):
    channel_id = message.channel.id
    thread_id = message.reference.message_id if message.reference else ''
    combo_id = get_combo_id(message)
    # check our cache
    if combo_id in channel_histories:
        return channel_histories[combo_id]
    else:
        # new bucket
        channel_histories[combo_id] = []
        if thread_id:
            # get the thread history
            # get the last 20 messages from the thread
            channel = await client.fetch_channel(channel_id)
            thread = await channel.fetch_message(thread_id)
            async for message in thread.history(limit=20):
                channel_histories[combo_id].append({
                    "role": "user" if message.author == client.user else "assistant",
                    "content": message.content
                })
        else:
            # get the channel history
            # get the last 20 messages from the channel
            async for message in message.channel.history(limit=20):
                channel_histories[combo_id].append({
                    "role": "user" if message.author == client.user else "assistant",
                    "content": message.content
                })

        logging.debug(f"channel or thread history not found in cache, fetched {len(channel_histories[combo_id])} from discord")
        return channel_histories[combo_id]


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')
    # Set the bot's status to online when connected
    await client.change_presence(status=discord.Status.online)

@client.event
async def on_message(message):
    # warm up the cache
    channel_history = await get_thread_or_channel_history(message)
    combo_id = get_combo_id(message)
    if channel_history is None:
        # create a new history
        channel_history = []
        channel_histories[combo_id] = channel_history

    # list the keys in the cache
    print("channel history cache keys: ", channel_histories.keys())

    keywords = ['!codex', '!turbo', '!gpt4']
    will_use = next(filter(lambda x: x in message.content, keywords), None)

    message_content = will_use + ": " + message.content.strip()

    # push into running message history
    channel_history.append({
        "role": "user" if message.author == client.user else "assistant",
        "content": message_content
    })

    # cap the history at 20 messages
    if len(channel_history) > 20:
        channel_history.pop(0)

    if message.author == client.user:
        return

    print("on message: ", message.content)
    reply = f"No reply from OpenAI. for ${will_use}"

    if will_use == "!codex":
        prompt = message.content[6:]
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )
        reply = response.choices[0].text.strip()
    elif will_use == "!turbo":
        prompt = message.content.strip()
        messages = channel_history.copy()
        messages = messages + [
            {"role": "assistant", "content": "remember, i am a ChatGPT bot named Turbo in the PoweredOnSoftware discord server. i am quirky and fun, and my responses always have a little bit of dark or dry humor."},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            #prompt=prompt,
            # temperature=0.7,
            # max_tokens=256,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0
            messages=messages
        )
        reply = response.choices[0].message.content.strip()
    elif will_use == "!gpt4":
        prompt = message.content[4:].strip()
        print("prompt: ", prompt)
        _channel_history = channel_history.copy()
        # _channel_history = _channel_history + [
        #     {"role": "system", "content": "you are a ChatGPT bot named F4 in the PoweredOnSoftware discord server. your personality is a mix of a friendly and helpful bot, and a sarcastic and witty human.please cite previous messages when responding to a user."},
        # ]
        _count_self_tokens = count_self_tokens()

        _channel_history = _channel_history + [
            {"role": "assistant", "content": f"i am a discord bot, and i'm executed by discord_bot.py that file has {_count_self_tokens} tokens in it"},
        ]

        #prompt_full = "\n".join([f"{m['role']}: {m['content']}) " for m in _channel_history])
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=_channel_history,
        )
        reply = response.choices[0].message.content.strip()

    logging.info(reply)
    await message.channel.send(f"{will_use}: "+reply)

async def main():
    print("hi from main")
    path = os.path.dirname(os.path.realpath(__file__))
    print("log path: ", path)

    # do we need to strip trailing slash?
    if path.endswith("/"):
        path = path[:-1]

    # Configure logging to write messages to a file
    # set to the lowest level to capture all messages
    logging.basicConfig(filename=f'{path}/my_bot.out', level=logging.DEBUG)

    # Redirect standard output and error messages to the log file
    sys.stdout = open(f'{path}/my_bot.out', 'w')
    sys.stderr = open(f'{path}/my_bot.out', 'w')

    # Use shutdown_callback() as the shutdown callback
    #daemonocle.shutdown_callback(shutdown_callback)
    #logging.info(f"Bot process ID: {daemonocle.get_pid()}")
    await client.start(TOKEN)

# You can now use `@bot.tree.command()` as a decorator:
@client.tree.command()
async def my_command(interaction: discord.Interaction) -> None:
  await interaction.response.send_message("Hello from my command!")
### NOTE: the above is a global command, see the `main()` func below:

if __name__ == "__main__":
    # Use daemonocle to run the script as a daemon
    #daemon = daemonocle.Daemon(worker=main)
    #daemon.do_action('start')
    asyncio.run(main())