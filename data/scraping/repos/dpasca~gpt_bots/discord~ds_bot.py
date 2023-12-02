#==================================================================
# Created by Davide Pasca - 2023/09/16
#==================================================================

import os
import re
import discord
from discord.ext import commands, tasks
import openai
from collections import defaultdict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize the OpenAI API
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the bot
intents = discord.Intents.default()
intents.guilds = True
intents.guild_messages = True
intents.message_content = True
intents.members = True

client = commands.Bot(command_prefix="!", intents=intents)

# Constants
IGNORE_PREFIX = "!"
CHANNELS = ['general', 'test0', 'test1', 'test2', 'test3']
SYSTEM_PROMPT_CHARACTER = """You are a skillful highly logical assistant that
goes straight to the point, with a tiny bit of occasional sarcasm."""
SYSTEM_PROMPT_FIXED_FORMAT = """You are operating in a forum, where multiple users can interact with you.
Most messages will include a header (metadata) at the start with the format
$$HEADER_BEGIN$$ CURTIME:<timestamp>, FROM:<username>, TO:<username>, $$HEADER_END$$
Additional fields may be present in the header for added context.
Never generate the header yourself.
Given the context, you should determine if you need to reply to a message.
You should also determine if a message should have a direct mention to a user,
to resolve any ambiguity, like when other users are involved in the discussion.
When mentioning a user, use its plain name, do not use metadata format outside of the header.
If you don't wish to reply to a message, just produce empty content."""

# Member count tracking
guildMemberCounts = defaultdict(int)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}!")

    for guild in client.guilds:
        try:
            guildMemberCounts[guild.id] = guild.member_count
            print(f"Fetched {guild.member_count} members for guild {guild.name}")
        except Exception as e:
            print(f"Failed to fetch members for guild {guild.name}: {e}")

@client.event
async def on_member_join(member):
    guildMemberCounts[member.guild.id] += 1

@client.event
async def on_member_remove(member):
    guildMemberCounts[member.guild.id] = max(0, guildMemberCounts[member.guild.id] - 1)

# Utility function to clean username
def doCleanUsername(username):
    return username.replace(" ", "_").replace(r"[^\w\s]", "")

@client.event
async def on_message(message):
    # Debugging
    print(f"Debug: guildMemberCounts {dict(guildMemberCounts)}")

    if message.author.bot:
        return

    if message.channel.name not in CHANNELS:
        return

    if message.content.startswith(IGNORE_PREFIX) and not message.author.id == client.user.id:
        return

    async with message.channel.typing():
        pass

    conversation = [
        {"role": "system", "content": f"{SYSTEM_PROMPT_CHARACTER}\n{SYSTEM_PROMPT_FIXED_FORMAT}"}
    ]

    last_messages = [msg async for msg in message.channel.history(limit=10)]

    for msg in reversed(last_messages):
        timestampField = msg.created_at.isoformat()
        fromField = doCleanUsername(msg.author.name)
        toField = ""

        if msg.content.startswith(f"<@!{client.user.id}>"):
            toField = doCleanUsername(client.user.name)

        finalContent = f"$$HEADER_BEGIN$$ CURTIME:{timestampField}, FROM:{fromField},"
        if toField:
            finalContent += f" TO:{toField},"
        finalContent += " $$HEADER_END$$"
        finalContent += f" {msg.content}"

        role = "assistant" if msg.author.id == client.user.id else "user"

        conversation.append({"role": role, "content": finalContent})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=conversation
        )
    except Exception as e:
        print(f"OpenAI API Error: {e}")


    cleanResponseMsg = re.sub(
        r"\$\$HEADER_BEGIN\$\$.*?\$\$HEADER_END\$\$",
        "",
        response['choices'][0]['message']['content'])

    chunkSize = 2000  # Discord message character limit
    shouldMentionUser = False

    for i in range(0, len(cleanResponseMsg), chunkSize):
        chunk = cleanResponseMsg[i:i + chunkSize]
        replyText = f"<@{message.author.id}> {chunk}" if shouldMentionUser else chunk
        await message.channel.send(replyText)

# Run the bot
client.run(os.getenv("DISCORD_TOKEN"))

