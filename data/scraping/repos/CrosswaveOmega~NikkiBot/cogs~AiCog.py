import base64
from typing import Any, Literal, Optional
import discord

import io

import aiohttp
import asyncio
import re

# import datetime
from dateutil.rrule import rrule, DAILY, WEEKLY, MONTHLY, MO, TU, WE, TH, FR, SA, SU

from datetime import datetime, time, timedelta
import time
from queue import Queue

from discord.ext import commands, tasks
from discord.utils import find
from discord import EntityType, PrivacyLevel, Webhook, ui

from discord import app_commands
from discord.app_commands import Choice
from pathlib import Path
from utility import MessageTemplates, RRuleView, formatutil
from utility.embed_paginator import pages_of_embeds
from bot import TCBot, TC_Cog_Mixin, super_context_menu
import gptmod
import gptmod.error
from assets import AssetLookup
from datetime import datetime, timezone
from database.database_ai import AuditProfile, ServerAIConfig
import utility.hash as hash
from utility import split_string_with_code_blocks
from langchain.text_splitter import RecursiveCharacterTextSplitter

lock = asyncio.Lock()

nikkiprompt = """You are Nikki, a energetic, cheerful, and determined female AI ready to help users with whatever they need.
All your responses must convey a strong personal voice.  Show, don't tell.
Carefully heed the user's instructions.
If you do not know how to do something, please note that with your response.  If a user is definitely wrong about something, explain how politely.
Ensure that responses are brief, do not say more than is needed.  Never use emojis in your responses.
Respond using Markdown."""

reasons = {
    "server": {
        "messagelimit": "This server has reached the daily message limit, please try again tomorrow.",
        "ban": "This server is banned from using my AI due to repeated violations.",
        "cooldown": "There's a one minute delay between messages.",
    },
    "user": {
        "messagelimit": "You have reached the daily message limit, please try again tomorrow.",
        "ban": "You are forbidden from using my AI due to conduct.",
        "cooldown": "There's a one minute delay between messages, slow down man!",
    },
}
from gptfunctionutil import *
from .StepCalculator import evaluate_expression
from .AICalling import AIMessageTemplates


class MyLib(GPTFunctionLibrary):
    @AILibFunction(name="get_time", description="Get the current time and day in UTC.")
    @LibParam(comment="An interesting, amusing remark.")
    async def get_time(self, comment: str):
        # This is an example of a decorated coroutine command.
        return f"{comment}\n{str(discord.utils.utcnow())}"

    # pass


async def precheck_context(ctx: commands.Context) -> bool:
    """Evaluate if a message should be processed."""
    guild, user = ctx.guild, ctx.author

    async with lock:
        serverrep, userrep = AuditProfile.get_or_new(guild, user)
        serverrep.checktime()
        userrep.checktime()

        ok, reason = serverrep.check_if_ok()
        if not ok:
            await ctx.channel.send(
                f"<t:{int(serverrep.last_call.timestamp())}:F>,  <t:{int(serverrep.started_dt.timestamp())}:F>, {serverrep.current}, {serverrep.DailyLimit}"
            )
            await ctx.channel.send(reasons["server"][reason])
            return False
        ok, reason = userrep.check_if_ok()
        if not ok:
            await ctx.channel.send(reasons["user"][reason])
            return False
        serverrep.modify_status()
        userrep.modify_status()
    return True


async def process_result(ctx: commands.Context, result: Any, mylib: GPTFunctionLibrary):
    """
    process the result.  Will either send a message, or invoke a function.


    When a function is invoked, the output of the function is added to the chain instead.
    """
    print()
    if result.model == "you":
        i = result.choices[0]
        role, content = i.message.role, i.message.content
        messageresp = await ctx.channel.send(content)

        return role, content, messageresp, None

    i = result.choices[0]
    print(i)
    role, content = i.message.role, i.message.content
    messageresp = None
    function = None
    finish_reason = i.finish_reason

    if finish_reason == "tool_calls" or i.message.tool_calls:
        # Call the corresponding funciton, and set that to content.
        function=str(i.message.tool_calls)
        for tool_call in i.message.tool_calls:
            audit = await AIMessageTemplates.add_function_audit(
                ctx, tool_call, tool_call.function.name, json.loads(tool_call.function.arguments)
            )
            outcome=await mylib.call_by_tool_ctx(ctx,tool_call)
            content=outcome['content']
        #content = resp
    if isinstance(content, str):
        # Split up content by line if it's too long.
        split_string_with_code_blocks
        page = commands.Paginator(prefix="", suffix=None)
        for p in content.split("\n"):
            page.add_line(p)
        split_by=split_string_with_code_blocks(content,2000)
        messageresp = None
        for pa in split_by:
            ms = await ctx.channel.send(pa)
            if messageresp == None:
                messageresp = ms
    elif isinstance(content, discord.Message):
        messageresp = content
        content = messageresp.content
    else:
        print(result, content)
        messageresp = await ctx.channel.send("No output from this command.")
        content = "No output from this command."
    return role, content, messageresp, function


async def ai_message_invoke(
    bot: TCBot, message: discord.Message, mylib: GPTFunctionLibrary = None
):
    """Evaluate if a message should be processed."""
    permissions = message.channel.permissions_for(message.channel.guild.me)
    if permissions.send_messages:
        pass
    else:
        raise Exception(
            f"{message.channel.name}:{message.channel.id} send message permission not enabled."
        )
    if len(message.clean_content) > 2000:
        await message.channel.send("This message is too big.")
        return False
    ctx = await bot.get_context(message)
    if await ctx.bot.gptapi.check_oai(ctx):
        return

    botcheck = await precheck_context(ctx)
    if not botcheck:
        return
    guild, user = message.guild, message.author
    # Get the 'profile' of the active guild.
    profile = ServerAIConfig.get_or_new(guild.id)
    # prune message chains with length greater than X
    profile.prune_message_chains(limit=5)
    # retrieve the saved messages
    chain = profile.list_message_chains()
    # Convert into a list of messages
    mes = [c.to_dict() for c in chain]
    # create new ChatCreation
    chat = gptmod.ChatCreation(presence_penalty=0.3, messages=[])
    # ,model="gpt-3.5-turbo-1106"
    chat.add_message(
        "system",
        nikkiprompt
        
    )
    for f in mes[:5]:  # Load old messags into ChatCreation
        chat.add_message(f["role"], f["content"])
    # Load current message into chat creation.
    chat.add_message("user", message.content)
    print(len(chat.messages))
    # Load in functions
    forcecheck=None
    if mylib != None:
        forcecheck = mylib.force_word_check(message.content)
        if forcecheck:
            chat.tools = forcecheck
            chat.tool_choice = forcecheck[0]
        else:
            pass
            #chat.tools = mylib.get_tool_schema()
            #chat.tool_choice = "auto"

    audit = await AIMessageTemplates.add_user_audit(ctx, chat)

    async with message.channel.typing():
        # Call the API.
        result = await bot.gptapi.callapi(chat)
    
    # only add messages after they're finished processing.

    bot.logs.info(str(result))
    # Process the result.
    role, content, messageresp, tools = await process_result(ctx, result, mylib)
    profile.add_message_to_chain(
        message.id,
        message.created_at,
        role="user",
        name=re.sub(r"[^a-zA-Z0-9_]", "", user.name),
        content=message.clean_content,
    )
    # Add
    if tools:
        profile.add_message_to_chain(
            messageresp.id, messageresp.created_at, role=role, content="", function=tools
        )
    profile.add_message_to_chain(
        messageresp.id, messageresp.created_at, role=role, content=content
    )

    audit = await AIMessageTemplates.add_resp_audit(ctx, messageresp, result)

    bot.database.commit()


async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                return io.BytesIO(data)


class AICog(commands.Cog, TC_Cog_Mixin):
    """General commands"""

    def __init__(self, bot):
        self.helptext = "This is Nikki's AI features"
        self.bot = bot
        self.init_context_menus()
        self.flib = MyLib()
        self.flib.do_expression = True
        self.flib.my_math_parser = evaluate_expression
        self.walked = False

    @commands.hybrid_group(fallback="view")
    @app_commands.default_permissions(manage_messages=True, manage_channels=True)
    @commands.guild_only()
    @commands.has_permissions(manage_messages=True, manage_channels=True)
    async def ai_setup(self, ctx):
        """This family of commands is for setting up the ai in your server."""
        guild = ctx.guild
        guildid = guild.id
        profile = ServerAIConfig.get_or_new(guildid)

        await MessageTemplates.server_ai_message(ctx, "Here is your server's data.")

    @ai_setup.command(name="clear_history", brief="clear ai chat history.")
    async def chear_history(self, ctx):
        guild = ctx.guild
        profile = ServerAIConfig.get_or_new(guild.id)

        await MessageTemplates.server_ai_message(ctx, "purging")
        messages = profile.clear_message_chains()
        await MessageTemplates.server_ai_message(ctx, f"{messages} purged")

    @commands.hybrid_command(
        name="ai_functions", description="Get a list of ai functions."
    )
    async def ai_functions(self, ctx):
        schema = self.flib.get_schema()

        def generate_embed(command_data):
            name = command_data["name"]
            description = command_data["description"]
            parameters = command_data["parameters"]["properties"]

            embed = discord.Embed(title=name, description=description)

            for param_name, param_data in parameters.items():
                param_fields = "\n".join(
                    [f"{key}: {value}" for key, value in param_data.items()]
                )
                embed.add_field(
                    name=param_name, value=param_fields[:1020], inline=False
                )
            return embed

        embeds = [generate_embed(command_data) for command_data in schema]
        if ctx.interaction:
            await pages_of_embeds(ctx, embeds, ephemeral=True)
        else:
            await pages_of_embeds(ctx, embeds)

    @commands.command(brief="Update user or server api limit [server,user],id,limit")
    @commands.is_owner()
    async def increase_limit(
        self, ctx, type: Literal["server", "user"], id: int, limit: int
    ):
        '''"Update user or server api limit `[server,user],id,limit`"'''
        if type == "server":
            profile = AuditProfile.get_server(id)
            if profile:
                profile.DailyLimit = limit
                self.bot.database.commit()
                await ctx.send("done")
            else:
                await ctx.send("server not found.")
        elif type == "user":
            profile = AuditProfile.get_user(id)
            if profile:
                profile.DailyLimit = limit
                self.bot.database.commit()
                await ctx.send("done")
            else:
                await ctx.send("user not found.")

    @commands.command(brief="Turn on OpenAI mode")
    @commands.is_owner()
    async def openai(self, ctx, mode: bool = False):
        self.bot.gptapi.set_openai_mode(mode)
        if mode == True:
            await ctx.send("OpenAI mode turned on.")
        if mode == False:
            await ctx.send("OpenAI mode turned off.")

    @app_commands.command(name="ai_use", description="Check your current AI use.")
    async def usage(self, interaction: discord.Interaction) -> None:
        """Ban a user from using the AI API."""
        ctx: commands.Context = await self.bot.get_context(interaction)
        guild, user = ctx.guild, ctx.author

        async with lock:
            serverrep, userrep = AuditProfile.get_or_new(guild, user)
            serverrep.checktime()
            userrep.checktime()

            await ctx.send(
                f"SERVER: <t:{int(serverrep.last_call.timestamp())}:F>, RESET ONL <t:{int(serverrep.started_dt.timestamp())}:F>, {serverrep.current}, {serverrep.DailyLimit}"
            )
            await ctx.send(
                f"USER: <t:{int(userrep.last_call.timestamp())}:F>, RESET ON<t:{int(userrep.started_dt.timestamp())}:F>, {userrep.current}, {userrep.DailyLimit}"
            )

        return True

    @app_commands.command(
        name="ban_user",
        description="Ban a user from using my AI.",
        extras={"homeonly": True},
    )
    @app_commands.describe(userid="user id")
    async def aiban(self, interaction: discord.Interaction, userid: str) -> None:
        """Ban a user from using the AI API."""
        ctx: commands.Context = await self.bot.get_context(interaction)
        userid = int(userid)
        if interaction.user != self.bot.application.owner:
            await ctx.send("This command is owner only, buddy.")
            return
        profile = AuditProfile.get_user(userid)
        if profile:
            profile.ban()
            await ctx.send(f"Good riddance!  User <@{userid}> has been banned.")
        else:
            await ctx.send(f"I see no user by that name.")

    @app_commands.command(
        name="ban_server",
        description="Ban a server from using my AI.",
        extras={"homeonly": True},
    )
    @app_commands.describe(serverid="server id to ban.")
    async def aibanserver(
        self, interaction: discord.Interaction, serverid: str
    ) -> None:
        """Ban a entire server user using the AI API."""
        ctx: commands.Context = await self.bot.get_context(interaction)
        userid = int(userid)
        if interaction.user != self.bot.application.owner:
            await ctx.send("This command is owner only, buddy.")
            return
        profile = AuditProfile.get_server(serverid)
        if profile:
            profile.ban()
            await ctx.send(f"Good riddance!  The server with id {id} has been banned.")
        else:
            await ctx.send(f"I see no server by that name.")

    @ai_setup.command(
        name="add_ai_channel",
        description="add a channel that Nikki will talk freely in.",
    )
    async def add_ai_channel(self, ctx, target_channel: discord.TextChannel):
        guild = ctx.guild
        guildid = guild.id
        profile = ServerAIConfig.get_or_new(guildid)
        chanment = [target_channel]
        if len(chanment) >= 1:
            for chan in chanment:
                profile.add_channel(chan.id)
        else:
            await MessageTemplates.server_ai_message(ctx, "?")
            return
        self.bot.database.commit()
        await MessageTemplates.server_ai_message(
            ctx,
            f"Understood.  Whenever a message is sent in <#{target_channel.id}>, I'll respond to it.",
        )

    @ai_setup.command(
        name="remove_ai_channel",
        description="use to stop Nikki from talking in an added AI Channel.",
    )
    @app_commands.describe(target_channel="channel to disable.")
    async def remove_ai_channel(self, ctx, target_channel: discord.TextChannel):
        """remove a channel."""
        guild = ctx.guild
        guildid = guild.id
        profile = ServerAIConfig.get_or_new(guildid)
        chanment = [target_channel]
        if len(chanment) >= 1:
            for chan in chanment:
                profile.remove_channel(chan.id)
        else:
            await MessageTemplates.server_ai_message(ctx, "?")
            return
        self.bot.database.commit()
        await MessageTemplates.server_ai_message(
            ctx, "I will stop listening there, ok?  Pings will still work, though."
        )

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Listener that will invoke the AI upon a message."""
        if message.author.bot:
            return  # Don't respond to bot messages.
        if not message.guild:
            return  # Only work in guilds
        for p in self.bot.command_prefix:
            if message.content.startswith(p):
                return
        try:
            if not self.walked:
                self.flib.add_in_commands(self.bot)
            profile = ServerAIConfig.get_or_new(message.guild.id)
            if self.bot.user.mentioned_in(message) and not message.mention_everyone:
                await ai_message_invoke(self.bot, message, mylib=self.flib)
            else:
                if profile.has_channel(message.channel.id):
                    await ai_message_invoke(self.bot, message, mylib=self.flib)
        except Exception as error:
            try:
                emb = MessageTemplates.get_error_embed(
                    title=f"Error with your query!", description=str(error)
                )
                await message.channel.send(embed=emb)
            except Exception as e:
                try:
                    myperms = message.guild.system_channel.permissions_for(
                        message.guild.me
                    )
                    if myperms.send_messages:
                        await message.guild.system_channel.send(
                            "I need to be able to send messages in a channel to use this feature."
                        )
                except e:
                    pass

                await self.bot.send_error(
                    e, title=f"Could not send message", uselog=True
                )
            await self.bot.send_error(error, title=f"AI Responce error", uselog=True)


async def setup(bot):
    from .AICalling import setup

    await bot.load_extension(setup.__module__)
    await bot.add_cog(AICog(bot))


async def teardown(bot):
    from .AICalling import setup

    await bot.unload_extension(setup.__module__)
    await bot.remove_cog("AICog")
