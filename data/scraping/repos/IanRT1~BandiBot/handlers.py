import discord
import re
from textwrap import dedent
import pytz
import logging
import json
from utils import (
    clean_username,
    get_current_pst_time,
    get_current_pst_date,
    get_server_info,
)
from openai_utils import send_to_openai

# Function to build context information for conversation
def build_context_info(message, categories, client):
    # Extract relevant information from the message and server
    user_nick_or_name = clean_username(message.author.nick, message.author.name)
    server_name = message.guild.name
    channel_name = message.channel.name
    bot_display_name = client.user.display_name
    prompt_start = message.content.split()[0]
    user_message = f"{user_nick_or_name}: {message.content.replace(prompt_start, '').lstrip()}"
    creation_date = message.guild.created_at.strftime("%Y-%m-%d")
    server_owner = clean_username(message.guild.owner.nick, message.guild.owner.name)
    total_members = message.guild.member_count
    current_pst_time = get_current_pst_time()
    current_pst_date = get_current_pst_date()
    server_info = get_server_info(message.guild)
    online_members_count = server_info["online_count"]

    # List of online members' nicknames/usernames
    online_users_list = ", ".join([info[0] for info in server_info["online_members"]])

    # Construct context information
    context = dedent(
        f"""
    **Server Information**:
    - Server Name: {server_name}
    - Active Channel: {channel_name}
    - Bot Name: {bot_display_name}
    - Server Creation Date: {creation_date}
    - Server Owner: {server_owner}
    - Total Members: {total_members}
    - Online Members: {online_members_count}
    - Online Users: {online_users_list}
    - Current Date: {current_pst_date}
    - Server Time: {current_pst_time}
    """
    ).strip()

    # Add member activity context if applicable
    if "Member Activity" in categories:
        context += build_member_activity_context(message)

    return context, user_message


# Function to build context information about member activities
def build_member_activity_context(message):
    # Get server information
    server_info = get_server_info(message.guild)

    # Create a table of online members
    online_table = (
        "\n".join(
            [
                f"{m[0]} - Roles: {', '.join(m[1])} - Joined {m[2]} days ago - Permissions: {' '.join(m[3])}"
                for m in server_info["online_members"]
            ]
        )
        or "No one"
    )

    # Create a list of members playing games
    playing_info = (
        ", ".join([f"{m[0]} is playing {m[1]}" for m in server_info["members_playing"]])
        or "No one is playing any games"
    )

    # Create a list of members in voice chat channels
    voice_chat_info_list = []
    for vc_name, members in server_info["voice_channels_info"].items():
        formatted_vc_members = ", ".join(members)
        voice_chat_info_list.append(f"In '{vc_name}': {formatted_vc_members}")
    voice_chat_info = "; ".join(voice_chat_info_list) or "No one is in voice chat"

    # Construct member activity context
    context_info = f"""
    **Member Activities**:
    - Online Members:
    {online_table}
    - Current Activities: {playing_info}
    - Members in Voice Chat: {voice_chat_info}
    """

    return context_info


# Function to build special instructions for the bot
def build_instruction(categories, bot_display_name, server_name):

    # Load the config.json data
    with open("config.json", "r") as file:
        config_data = json.load(file)

    instruction = config_data["instructions"]["initial"]
    special_instructions = config_data["special_instructions"]

    # Add special instructions for categories if applicable
    for category in categories:
        if category in special_instructions:
            instruction += f"\nThere are your special instructions for this message: {special_instructions[category]}"

    return instruction


# Asynchronous function to handle bot mentions
async def handle_bot_mention(message, categories, client):
    # Get the bot's display name
    bot_display_name = client.user.display_name

    # Fetch recent messages from the channel
    recent_messages_str = await fetch_recent_messages(message.channel, limit=20)

    # Build the context and instruction once
    context_info, user_message = build_context_info(message, categories, client)
    instruction = build_instruction(categories, bot_display_name, message.guild.name)

    # Create the payload for OpenAI
    conversation_payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "system", "content": context_info},
            {"role": "system", "content": recent_messages_str},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.5,
    }

    # Send the payload to OpenAI and process the response
    response_data = await send_to_openai(conversation_payload)
    if not response_data:
        await message.channel.send(
            f"{message.author.mention} Sorry, I encountered an issue processing your request. Please try again later."
        )
        return

    response_text = process_openai_response(data=response_data, message=message, client=client)

    # Send the response to the channel
    await send_response_to_channel(message, response_text)


# Asynchronous function to fetch recent messages from a channel
async def fetch_recent_messages(channel, limit=20):
    # Fetch the most recent messages from the channel
    messages = [msg async for msg in channel.history(limit=limit)]
    # Reverse the messages so they are in chronological order
    messages.reverse()

    # Format the messages
    formatted_messages = []
    for msg in messages:
        # Check if the author is a member of the guild
        if isinstance(msg.author, discord.Member):
            cleaned_name = clean_username(msg.author.nick, msg.author.name)
        else:
            cleaned_name = msg.author.name

        # Get the timestamp of the message in PST and format it
        pacific = pytz.timezone("US/Pacific")
        timestamp = msg.created_at.astimezone(pacific).strftime("%I:%M %p")

        content = msg.content
        for member in channel.guild.members:
            member_name = clean_username(member.nick, member.name)
            content = content.replace(member.mention, member_name)

        formatted_messages.append(f"[{timestamp}] {cleaned_name}: {content}")

    return "\n".join(formatted_messages)


# Function to process OpenAI's response
def process_openai_response(data, message, client):
    if "choices" not in data:
        logging.error(f"Unexpected OpenAI API response: {data}")
        return "Sorry, I encountered an issue processing your request."

    response_text = data["choices"][0]["message"]["content"].strip()

    bot_display_name = client.user.display_name
    user_name = clean_username(message.author.nick, message.author.name)

    # Patterns to look for: bot's name followed by a colon,
    # timestamp followed by bot's name and a colon,
    # and the nickname or username of the prompter followed by a quote.
    patterns_to_strip = [
        re.compile(r"^" + re.escape(bot_display_name) + r":"),
        re.compile(r"^\[\d{1,2}:\d{2} (AM|PM)\] " + re.escape(bot_display_name) + r":"),
        re.compile(r"^" + re.escape(user_name) + r"[:,]"),
    ]

    for pattern in patterns_to_strip:
        if pattern.match(response_text):
            response_text = pattern.sub("", response_text).strip()

    return response_text


# Asynchronous function to send the response to the channel
async def send_response_to_channel(message, response_text):
    try:
        await message.channel.send(f"{message.author.mention} {response_text}")
    except Exception as e:
        logging.error(f"Error sending message: {e}")
        await message.channel.send(
            f"{message.author.mention} An error occurred. Please try again..."
        )
