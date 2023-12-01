import datetime
from typing import Any
import discord
import traceback
from urllib.request import Request, urlopen
import asyncio
import configparser
import os
from pathlib import Path

import logging
import logging.handlers

import sys, random


from queue import Queue


from utility import serverOwner, serverAdmin, MessageTemplates, formatutil

from discord.ext import commands, tasks

from discord import Webhook
from subprocess import Popen

import gui

from collections import defaultdict
from dateutil.rrule import rrule, rrulestr, WEEKLY, SU
from .TauCetiBot import TCBot, ConfigParserSub
from .Tasks.TCTasks import TCTask, TCTaskManager
from .TcGuildTaskDB import TCGuildTask
from .TCAppCommandAutoSync import AppGuildTreeSync
from .errorformat import client_error_message
from .config_gen import setup, config_update

"""
Initalizes TCBot, defines some checks, and contains the main setup coroutine.

"""

import database.database_main as dbmain
from assets import AssetLookup
import openai


bot: TCBot = TCBot()


taskflags = defaultdict(bool)


async def opening():
    gui.gprint("OK.")


@bot.check
async def is_cog_enabled(ctx: commands.Context):
    return True


@bot.check
async def user_wants_ignore(ctx: commands.Context):
    """ignore commands if the user doesn't want them."""
    uid = ctx.author.id
    if dbmain.Users_DoNotTrack.check_entry(uid):
        return False
    return True


@bot.check
async def guildcheck(ctx):
    if ctx.guild != None:
        serverdata = dbmain.ServerData.get_or_new(ctx.guild.id)
        serverdata.update_last_time()
        if ctx.command.extras:
            if "guildtask" in ctx.command.extras and ctx.guild != None:
                if taskflags[str(ctx.guild.id)]:
                    return False
    return True


@bot.on_error
def on_error(event_method: str, /, *args: Any, **kwargs: Any):
    gui.gprint("Error?")
    log = logging.getLogger("discord")
    log.exception("Ignoring exception AS in %s", event_method)
    try:
        just_the_string = "".join(traceback.format_exc())
        er = MessageTemplates.get_error_embed(
            title=f"Error with {event_method}", description=f"{just_the_string}"
        )
        asyncio.create_task(bot.send_error_embed(er))
    except:
        gui.gprint("The error had an error.")


@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction, error: discord.app_commands.AppCommandError
):
    logger = logging.getLogger("discord")
    logger.error(
        "Ignoring exception in app command %s", interaction.command.name, exc_info=error
    )
    ctx = await bot.get_context(interaction)
    if "guildtask" in ctx.command.extras and ctx.guild != None:
        taskflags[str(ctx.guild.id)] = False
    await bot.send_error(error, f"App Command {interaction.command.name}")
    errormess = client_error_message(
        error, name=f"App Command {interaction.command.name}"
    )
    emb = MessageTemplates.get_error_embed(
        title=f"Error with {ctx.message.content}", description=f"{errormess}"
    )
    try:
        await ctx.send(embed=emb)
    except Exception as e:
        bot.logs.error(str(e))


@bot.event
async def on_command_error(ctx, error):
    """this function logs errors for prefix commands."""
    gui.gprint(error)
    log = logging.getLogger("discord")

    command: discord.ext.commands.command = ctx.command
    log.error("Ignoring exception in command %s", command, exc_info=error)
    errormess = "Command not found I think..."
    if ctx.command:
        errormess = client_error_message(error, name=f" Command {command.name}")
    else:
        errormess = client_error_message(error, name=f"{ctx.message.content}")
    gui.gprint(errormess)
    emb = MessageTemplates.get_error_embed(
        title=f"Error with {ctx.message.content}", description=f"{errormess}"
    )
    await bot.send_error(error, title=f"Error with {ctx.message.content}")
    try:
        await ctx.send(embed=emb)
    except Exception as e:
        bot.logs.error(str(e))


@bot.before_invoke
async def add_log(ctx):
    gui.gprint(f"{ctx.invoked_with},{str(ctx.invoked_parents)}")
    if ctx.command.extras:
        if "guildtask" in ctx.command.extras and ctx.guild != None:
            taskflags[str(ctx.guild.id)] = True

    gui.gprint(f"Firing {ctx.command.name}")


@bot.after_invoke
async def free_command(ctx):
    if "guildtask" in ctx.command.extras and ctx.guild != None:
        taskflags[str(ctx.guild.id)] = False

    if ctx.command_failed:
        await ctx.send(f"ERROR, {ctx.message.content} failed!  ")


@bot.event
async def on_app_command_completion(
    interaction: discord.Interaction, command: discord.app_commands.Command
):
    gui.gprint("app command complete: ", command.name)


@bot.event
async def on_connect():
    gui.gprint(f"{datetime.datetime.now()}: Bot connected.")


@bot.event
async def on_disconnect():
    gui.gprint(f"{datetime.datetime.now()}: Bot disconnected.")


@bot.event
async def on_ready():
    gui.gprint("Connection ready.")
    await opening()
    gui.gprint("BOT ACTIVE")
    try:
        if bot.error_channel == -726:
            # Time to make a new guild.
            from .guild_maker import new_guild

            await new_guild(bot)
    except Exception as e:
        gui.gprint(e)
        await bot.close()
    await bot.after_startup()
    gui.gprint("Setup done.")


class Main(commands.Cog):
    """debug class, only my owner can use these."""

    async def cog_check(self, ctx):
        if ctx.author.id == ctx.bot.application.owner.id:
            return True
        return False

    @commands.command(hidden=True)
    async def shutdown(self, ctx):
        """
        debug command, shuts the bot down.
        """
        bot = ctx.bot
        auth = ctx.message.author
        channel = ctx.message.channel

        bot.post_queue_message.cancel()
        bot.delete_queue_message.cancel()
        bot.check_tc_tasks.cancel()

        await ctx.channel.send("Shutting Down")

        await ctx.bot.close()

    @commands.command()
    async def task_view(self, ctx):
        """debugging only."""
        bot = ctx.bot
        guild = ctx.guild
        list = TCTaskManager.task_check()
        chunks = [", ".join(list[i : i + 10]) for i in range(0, len(list), 10)]
        formatted_strings = [chunk for chunk in chunks]
        for i in formatted_strings:
            await ctx.send(i)

    @commands.command()
    async def config_view(self, ctx):
        """view the config.ini file"""
        # bot=ctx.bot
        # guild=ctx.guild
        print(bot.config.values())
        listv = []
        for v in bot.config.values():
            for k, s in v.items():
                listv.append(f"{v.name}, {k}, `{s}`")
            print(listv)
        pages = commands.Paginator(prefix="", suffix="")
        for l in listv:
            pages.add_line(l)
        for p in pages.pages:
            await ctx.send(p)

    @commands.command()
    async def config_set(self, ctx, section: str, option: str, value: str):
        """set a value in the config.ini file"""
        # bot=ctx.bot
        # guild=ctx.guild
        print(bot.config.values())
        list = []
        for v in bot.config.values():
            for k, s in v.items():
                list.append(f"{v.name}, {k}, `{s}`")
            print(list)

        pages = commands.Paginator(prefix="", suffix="")
        for l in list:
            pages.add_line(l)
        for p in pages.pages:
            await ctx.send(p)
        if not ctx.bot.config.has_section(section):
            ctx.bot.config.add_section(section)

        ctx.bot.config.set(section, option, value)
        ctx.bot.config.write(open("config.ini", "w"))

    @commands.command()
    async def assetlookuptest(self, ctx):
        """debugging only."""
        bot = ctx.bot
        guild = ctx.guild
        AssetLookup.save_assets()
        names = AssetLookup.get_asset("blanknames")

    @commands.command()
    async def database_debug(self, ctx):
        """debugging only."""
        bot = ctx.bot
        guild = ctx.guild
        result = bot.database.database_check()
        pageme = commands.Paginator(prefix="", suffix="", max_size=4096)
        for p in result.split("\n"):
            pageme.add_line(p)
        embeds = []
        for page in pageme.pages:
            embed = discord.Embed(
                title="res", description=page, color=discord.Color(0xFF0000)
            )
            embed.set_author(name="DATABASE STATUS:")
            embeds.append(embed)
        for i in embeds:
            await ctx.send(embed=i)

    @commands.command()
    async def task_add(self, ctx):
        """debugging only."""
        bot = ctx.bot
        guild = ctx.guild
        message = await ctx.send("Target Message.")
        myurl = message.jump_url
        robj = rrule(freq=WEEKLY, byweekday=SU, dtstart=datetime(2023, 1, 1, 15, 0))
        new = TCGuildTask.add_guild_task(guild.id, "COMPILE", message, robj)
        new.to_task(bot)

    @commands.command()
    async def task_remove(self, ctx):
        """debugging only."""
        return

    @commands.command()
    async def ping_tester(self, ctx: commands.Context, content: str = "new"):
        """debugging only."""
        await ctx.send(content=content)

    @commands.command(hidden=True)
    async def purge_guild_data(self, ctx, guildid: int):
        """debugging only."""
        profile = dbmain.ServerData.get_or_new(guildid)
        if profile:
            profile.last_use = datetime.datetime.fromtimestamp(800000)
            ctx.bot.database.commit()
            await ctx.bot.audit_guilds(override_for=guildid)
            await ctx.send("Data purged.")
        return

    @commands.command(hidden=True)
    @commands.guild_only()
    async def do_not_sync(self, ctx):
        """debugging only."""
        profile = AppGuildTreeSync.get(ctx.guild.id)
        if profile:
            AppGuildTreeSync.setdonotsync(ctx.guild.id)
            await ctx.send(f"{profile.donotsync} Sync disabled.")
            ctx.bot.database.commit()
            ctx.bot.tree.clear_commands(guild=ctx.guild)
            await ctx.bot.tree.sync(guild=ctx.guild)
            synced = []
            await ctx.send("Sync disabled.")
        return

    @commands.command()
    async def reload(self, ctx):
        """debugging only."""
        bot = ctx.bot
        bot.update_ext_list()
        await bot.reload_all(True)
        embed = discord.Embed(title="Reloaded Loaded Extensions")
        for i, v in bot.loaded_extensions.items():
            ex, val = v
            embed.add_field(name=i, value=ex, inline=True)
            if val:
                exepemb = discord.Embed(title=f"Error={i}", description=f"{ex}\n{val}")
                await ctx.send(embed=exepemb)
            if len(embed.fields) > 20:
                await ctx.send(embed=embed)
                embed = discord.Embed(title="Loaded Extensions")
        await ctx.send(embed=embed)

    @commands.command()
    async def view_extend_status(self, ctx):
        """debugging only."""

        bot = ctx.bot
        embed = discord.Embed(title="Loaded Extensions")
        for i, v in bot.loaded_extensions.items():
            ex, val = v
            embed.add_field(name=i, value=ex, inline=True)
            if val:
                gui.gprint(f"{ex}\n{val}")
            if len(embed.fields) > 20:
                await ctx.send(embed=embed)
                embed = discord.Embed(title="Loaded Extensions")
        await ctx.send(embed=embed)

    @commands.command()
    async def reload_extend(self, ctx, extname: str):
        """debugging only."""

        bot = ctx.bot

        result = await bot.extension_loader(extname)
        if result == "NOTFOUND":
            await ctx.send("I don't have an extension by that name.")
        elif result == "LOADOK":
            await ctx.send("I reloaded the extension without problems.")
        else:
            embed = discord.Embed(title="Embed Error Extensions")
            v = bot.loaded_extensions[extname]
            ex, val = v
            if val:
                exepemb = discord.Embed(
                    title=f"Error={extname}", description=f"{ex}\n{val}"
                )
                await ctx.send(embed=exepemb)
            else:
                await ctx.send("...what?")


async def main(args):
    """setup and start the bot."""
    config, keys = setup(args)
    config_update(config)
    AssetLookup()
    if config == None or keys == None:
        return
    async with bot:
        intent = discord.Intents.default()
        intent.presences = True
        intent.message_content = True
        intent.guilds = True
        intent.members = True

        outcome = bot.set_error_channel(config.get("optional", "error_channel_id"))
        if not outcome:
            bot.error_channel = -726
        bot.config = config
        await bot.add_cog(Main())
        bot.set_ext_directory("./cogs")
        gui.DataStore.initialize("./saveData/ds.sqlite")
        gui.DataStore.initialize_default_values()

        if config != None:
            g = keys.get("optional", "google", fallback=None)
            c = keys.get("optional", "cse_id", fallback=None)
            bot.keys["google"] = g
            bot.keys["cse"] = c
            openai.api_key = keys.get("optional", "openai")

            await bot.start(keys.get("vital", "cipher"))
