import os
import openai
import discord
import re
import emoji
import asyncio
import aiohttp
from discord.ext import commands
from dotenv import load_dotenv
from datetime import datetime, timedelta

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD = os.getenv("DISCORD_GUILD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up bot with intents
intents = discord.Intents.all()
intents.messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Set up OpenAI API
openai.api_key = OPENAI_API_KEY

# Global variables
threshold = 2
cooldown_duration = 5  # default is 5 seconds
last_api_call_time = None
message_cache = {}
rate_limit_duration = timedelta(minutes=5)

# Add the on_guild_join event
@bot.event
async def on_guild_join(guild):
    owner = guild.owner

    if not OPENAI_API_KEY:
        message = (
            "Heyo! Thanks for adding me to your server. To get started, please provide your OpenAI API key by typing "
            "`!apikey YOUR_API_KEY` in this DM. Replace `YOUR_API_KEY` with your actual API key. "
            "I won't do anything until you provide me with an API key."
        )
        try:
            await owner.send(message)
        except Exception as e:
            print(f"Error sending DM to guild owner: {e}")

# Check if a message is cached and recent
async def is_message_cached_and_recent(message):
    now = datetime.now()
    if message.content in message_cache:
        last_seen = message_cache[message.content]
        if now - last_seen < rate_limit_duration:
            return True

    # Update the message_cache with the current timestamp
    message_cache[message.content] = now
    return False

# Get emojis and emotional weight for a given message
async def get_emojis_and_weight(message):
    if not OPENAI_API_KEY:
        await message.channel.send("OpenAI API key is not set. Please set it using the !apikey command in a DM to the bot.")
        return [], 0

    async with aiohttp.ClientSession() as session:
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Emilia, a zoomer Discord gamer anime girl. You love anime. You're a bit of a anarchist communist and love"
                        "feminism. You're also kinda random, crazy, and fairly unpredictable. You're"
                        "egotistical and you absolutely pwn noobs. You're also very hypersexual. Your job is to "
                        "suggest emojis based on the context"
                        "and emotion of a message."
                        "Given the message, determine the suitable emoji reaction that is interesting and/or funny "
                        "based on the context and emotion."
                        "Rate the emotional weight of the message on a scale from 0 to 5 (0 being the lowest "
                        "emotional weight, and 5 being the highest),"
                        "with 3 as the average emotional weight. If the weight is 0, do not output any emoji. "
                        "Be a little creative in your emoji, it should be funny. Give it your all, and go totally"
                        "bonkers. When you feel the need to react with words, you can react with letter emojis that"
                        "spell out words."
                        "Examples can include COCK, BASED, HUNGRY, etc. Do note that because of Discord's limitations,"
                        "you cannot repeat the same letter emoji, so be mindful of that."
                        "Strictly adhere to the following format, without printing anything else: "
                        "output the emojis followed by the weight number, like this: 😃1. "
                        "Do not deviate from the specified format. There should be no whitespace, and there "
                        "should be no extra text or punctuation in the output. If you cannot provide an emoji, "
                        "do not provide any output."
                    ),
                },
                {"role": "user", "content": message.content},
            ],
            "max_tokens": 8,
            "n": 1,
            "stop": None,
            "temperature": 0.9,
        }

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        url = "https://api.openai.com/v1/chat/completions"

        async with session.post(url, json=data, headers=headers) as resp:
            response = await resp.json()

        output_lines = response['choices'][0]['message']['content'].strip().split("\n")

    # Extract weight and emojis from the output
    weight_pattern = re.compile(r'(\d)$')
    weight_match = weight_pattern.search(output_lines[0])

    if weight_match:
        weight = int(weight_match.group(1))
        emoji_string = output_lines[0][:weight_match.start()]
    else:
        weight = 0
        emoji_string = output_lines[0]

    emojis = [e for e in emoji_string if e in emoji.EMOJI_DATA]

    return emojis, weight


@bot.event
async def on_message(message):
    global threshold
    global last_api_call_time

    if message.author == bot.user:
        return

    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
    else:
        # Check if the message is cached and recent
        if await is_message_cached_and_recent(message):
            return

    # Check if the API key is set
    if not OPENAI_API_KEY:
        await message.channel.send("OpenAI API key is not set. Please set it using the !apikey command in a DM to the bot.")
        return

    # Check if the cooldown period has passed since the last API call
    now = datetime.now()
    if last_api_call_time is not None and (now - last_api_call_time).total_seconds() < cooldown_duration:
        print("Skipping API call due to cooldown")
        return

    emojis, weight = await get_emojis_and_weight(message)

    print(f"Emojis: {emojis}")
    print(f"Weight: {weight}")

    emoji_reacted = False
    if weight >= (6 - threshold):
        for emoji_text in emojis:
            emoji_unicode = emoji.emojize(emoji_text)
            try:
                await message.add_reaction(emoji_unicode)
                await asyncio.sleep(0.5)  # Add a short delay between reactions
                emoji_reacted = True
            except discord.errors.HTTPException:
                print(f"Unknown or invalid emoji: {emoji_unicode}")
                continue

    # Update the last_api_call_time only if an emoji was successfully reacted
    if emoji_reacted:
        last_api_call_time = now


@bot.command(name="ratelimit")
async def set_rate_limit_command(ctx, new_rate_limit: int):
    global rate_limit_duration

    if 1 <= new_rate_limit <= 60:
        rate_limit_duration = timedelta(minutes=new_rate_limit)
        await ctx.send(f"Rate limit has been set to {new_rate_limit} minutes")
    else:
        await ctx.send("Rate limit must be between 1 and 60 minutes")

@bot.command(name="apikey")
async def set_api_key_command(ctx, new_api_key: str):
    global OPENAI_API_KEY

    # Check if the message was sent in a DM
    if isinstance(ctx.channel, discord.DMChannel):
        # Set the API key to the user-provided value
        OPENAI_API_KEY = new_api_key

        # Update the OpenAI API key used by the bot
        openai.api_key = OPENAI_API_KEY

        await ctx.send("OpenAI API key has been updated")
    else:
        await ctx.send("Please send this command in a private message (DM) to the bot")

@bot.command(name="threshold")
async def set_threshold_command(ctx, new_threshold: int):
    global threshold

    if 0 <= new_threshold <= 5:
        threshold = new_threshold
        await ctx.send(f"Threshold has been set to {new_threshold}")
    else:
        await ctx.send("Threshold must be between 0 and 5")

@bot.command(name="cooldown")
async def set_cooldown_command(ctx, new_cooldown: int):
    global cooldown_duration

    if new_cooldown >= 1:
        cooldown_duration = new_cooldown
        await ctx.send(f"Cooldown duration has been set to {new_cooldown} seconds")
    else:
        await ctx.send("Cooldown duration must be at least 1 second")



bot.run(TOKEN)
