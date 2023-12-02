from typing import Any, Literal, Optional
import discord
import asyncio


from discord.ext import commands, tasks

import openai

from gptfunctionutil import *


"""
This example is for invoking discord.py bot commands with this utility.
It's done much in the same way as coroutines.
"""


class ExampleCog(commands.Cog):
    """The cogs do not need to be in the same file."""

    def __init__(self, bot):
        self.bot = bot

    # You can decorate commands in any cog, since they're all accessable through the commands.Bot object.
    @AILibFunction(name="wait_for", description="Wait for a few seconds, then return.")
    @LibParam(targetuser="Number of seconds to wait for.")
    @commands.command()
    async def wait_for(self, ctx, towait: int):
        # Wait for a set period of time.
        await ctx.send("launcing waitfor.")
        await asyncio.sleep(towait)
        m = await ctx.send(f"waited for {towait}'!")
        return m

    # At this time, this only works for prefix commands, not slash/app commands.


class MyLib(GPTFunctionLibrary):
    @AILibFunction(name="get_time", description="Get the current time and day in UTC.")
    @LibParam(comment="An interesting, amusing remark.")
    async def get_time(self, comment: str):
        # This is an example of a decorated coroutine.
        return f"{comment}\n{str(discord.utils.utcnow())}"


class AICog(commands.Cog):
    """Basic code for invoking decorated dpy commands with the library.
    It's in it's own cog"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # Include reference to your GPTFunctionLibrary subclass.
        self.mylib = MyLib()
        # Walked indicates that you've previously loaded the bot commands
        # Into your GPTFunctionLibrary subclass.
        self.client = openai.AsyncClient()
        self.walked = False

    async def ai_message_invoke(self, message: discord.Message):
        """Get string from message content, send to ai api, and process functions if needed.."""
        bot = self.bot

        # While not required, it's a good idea to make sure that the bot has
        # permission to send messages in the channel.
        permissions = message.channel.permissions_for(message.channel.guild.me)
        if permissions.send_messages:
            pass
        else:
            raise Exception(f"{message.channel.name}:{message.channel.id} send message permission not enabled.")
        if len(message.clean_content) > 2000:
            # Nitro users can send messages longer than 2000 characters.
            raise Exception(f"This message is too big.")

        # You need to get a commands.Context object in order to process bot commands.
        ctx: commands.Context = await bot.get_context(message)

        async with message.channel.typing():
            # Call the API.
            completion = await self.client.chat.completions.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message.content},
                ],
                functions=self.mylib.get_schema(),
                function_call="auto",
            )
        oaimessage = completion.choices[0].message
        if "function_call" in oaimessage:
            # Process function call.
            # For bot commands, you need to use call_by_dict_ctx.

            result = await self.mylib.call_by_dict_ctx(ctx, oaimessage["function_call"])
            # If the result is a string, split it up.
            # if it's a message, just print the content.
            if isinstance(result, str):
                page = commands.Paginator(prefix="", suffix="")
                for p in result.split("\n"):
                    page.add_line(p)
                for pagetext in page.pages:
                    ms = await ctx.send(pagetext)
            elif isinstance(result, discord.Message):
                content = result.content
                print(result, content)
        else:
            # No function_call field found.
            page = commands.Paginator(prefix="", suffix="")
            for p in oaimessage["content"].split("\n"):
                page.add_line(p)
            for pagetext in page.pages:
                await ctx.send(pagetext)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Whenever the bot gets a message, it will fire this function.

        if message.author.bot:
            return  # Don't respond to bot messages.
        if not message.guild:
            return  # Only work in guilds.
        for p in self.bot.command_prefix:
            if message.content.startswith(p):
                # This will prevent the bot from invoking the AI
                # If it detects the command prefix.
                return
        try:
            if not self.walked:
                # Check for any decorated bot commands, and add them to
                # your subclass instance.
                self.mylib.add_in_commands(self.bot)
                self.walked = True
            await self.ai_message_invoke(message)
        except Exception as error:
            await message.channel.send(str(error))


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


# cogs_added = False
async def add_cogs(bot):
    await bot.add_cog(AICog(bot))
    await bot.add_cog(ExampleCog(bot))


@bot.event
async def on_ready():
    await add_cogs(bot)
    print("Connection ready.")


token = "your token here"
bot.run(token)
