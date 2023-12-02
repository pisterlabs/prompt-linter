import asyncio
import re

import discord
from discord.ext import commands
import openai
import datetime


# need to remake this class, it currently doesn't use the overwritten memberconverter making its selection of members
# not work as expected


class Nickname(commands.Cog):
    def __init__(self, bot, openai_api_key):
        self.bot = bot
        self.openai_api_key = openai_api_key
        print("Nickname cog initialized")

    def timeframe_to_seconds(self, timeframe):

        if not timeframe[:-1]:  # Check if the string is empty
            return None

        time_unit = timeframe[-1].lower()
        # turn this if else ladder into a dictionary
        if time_unit not in ["h", "d", "w", "m", "y"]:
            return None
        d = {"h": 3600, "d": 86400, "w": 604800, "m": 2592000, "y": 31536000}
        return int(timeframe[:-1]) * d[time_unit]

    async def get_messages_in_timeframe(self, channel, timeframe):
        timeframe_seconds = self.timeframe_to_seconds(timeframe)
        if timeframe_seconds is None:
            return []

        now = datetime.datetime.utcnow()
        start_time = now - datetime.timedelta(seconds=timeframe_seconds)

        messages = []
        async for message in channel.history(after=start_time):
            if message.author != self.bot.user:
                messages.append(message)

        return messages

    async def generate_nickname(self, message_text):

        if self.openai_api_key is None:
            return "what you think this is free?"

        prompt = f"Generate a creative and relevant Discord nickname based on the following user's message history:\n\n{message_text}\n\nNickname:"

        response = await asyncio.to_thread(
            openai.Completion.create,
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
        )

        nickname = response.choices[0].text.strip()
        return nickname

    @commands.command(name="generate_nickname")
    async def generate_nickname_command(self, ctx, input_string: str):
        # Extract the channel mention (if any) using a regular expression
        channel_pattern = re.compile(r"<#(\d+)>")
        channel_mention = channel_pattern.search(input_string)

        if channel_mention:
            channel_id = int(channel_mention.group(1))
            channel = ctx.guild.get_channel(channel_id)
            user_name = input_string[:channel_mention.start()].strip()
        else:
            channel = ctx.channel
            user_name = input_string.strip()

        if channel is None:
            await ctx.send(f"Channel not found.")
            return

        await ctx.send(f"Got you fam, I'm using {channel} ")

        user = None
        for member in ctx.guild.members:
            if (user_name in member.display_name) or (member.nick and user_name in member.nick):
                user = member
                break

        if user is None:
            await ctx.send(f"User '{user_name}' not found.")
            return

        messages = await self.get_messages_in_timeframe(channel, "1m")
        user_messages = [msg.content for msg in messages if msg.author == user]
        message_text = " ".join(user_messages)

        nickname = await self.generate_nickname(message_text)
        await ctx.send(f"{user.display_name}? more like {nickname}")
