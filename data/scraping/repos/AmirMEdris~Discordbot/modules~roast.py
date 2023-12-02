import discord
from discord.ext import commands
import openai
import datetime

from modules.membercvtr import DisplayNameMemberConverter


class Roast(commands.Cog):
    def __init__(self, bot, openai_api_key):
        self.bot = bot
        self.openai_api_key = openai_api_key
        print("Roast cog initialized")

    async def generate_roast(self, user_name, messages):
        prompt = f"Based on these messages from {user_name}, create a funny roast remember to exclude any " \
                 f"names mention and focus on topics:\n{messages}\nRoast:"

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        joke = response.choices[0].text.strip()
        return joke

    class DisplayNameMemberConverter(commands.MemberConverter):
        async def convert(self, ctx, argument):
            for member in ctx.guild.members:
                if member.display_name.lower() == argument.lower():
                    return member
            raise commands.MemberNotFound(argument)

    # @bot.command(
    #     name="my_first_command",
    #     description="This is the first command I made!",
    # )
    @commands.command(name='roast')
    async def roast(self, ctx, *, user_name: str):
        try:
            # await ctx.send([member.display_name for member in ctx.guild.members])
            user = await DisplayNameMemberConverter().convert(ctx, user_name)

        except commands.MemberNotFound:
            await ctx.send(f"User {user_name} not found.")
            return

        messages = []
        async for message in ctx.channel.history(limit=1000):
            if message.author == user:
                messages.append(message.content)

        if not messages:
            await ctx.send(f"I couldn't find any messages from {user.mention}.")
            return

        # Combine last 40 messages or less
        messages = "\n".join(messages[:40])

        joke = await self.generate_roast(user_name, messages)
        await ctx.send(joke)
