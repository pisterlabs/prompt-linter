import os
from datetime import datetime
from asyncio import TimeoutError

from discord import Game, TextChannel
from discord.ext import commands, tasks
from langchain.memory import ConversationBufferWindowMemory

from topicQueue import TopicQueue
from chain import create_aniketh_ai
from ext import admin_dashboard, cmd_error, info_msg, loop_status

UPDATE_WAIT = int(os.environ.get("UPDATE_WAIT", 18))

class AdminCog(commands.Cog):
    def __init__(
        self, 
        bot: commands.Bot, 
        topic_queue: TopicQueue,
        log_channel_id: int,
        confirmation_emote_id: int = 1136812895859134555 #:L_: emoji
    ) -> None:
        self.bot = bot
        self.topic_queue = topic_queue
        self.log_channel_id = log_channel_id
        self.confim_emote_id = confirmation_emote_id

        self.start_datetime = datetime.now()
        self.locked = False

    @property
    def log_channel(self):
        self.bot.get_channel(self.log_channel_id)

    @property
    def confim_emote(self):
        self.bot.get_channel(self.confim_emote_id)

    @tasks.loop(hours=UPDATE_WAIT)
    async def set_status(self):
        chain = create_aniketh_ai(ConversationBufferWindowMemory(return_messages=True))
        topic = self.topic_queue.pick_topic()
        message = chain.predict(user_message=f"Write a short (5-6 word) sentence on {topic}")
        game = Game(message)
        await self.bot.change_presence(activity=game)

    @commands.group(pass_context=True)
    async def admin(self, ctx):
        if not ctx.invoked_subcommand:
            embed = await admin_dashboard(self.bot, self.start_datetime)
            await ctx.send(embed=embed)

    @admin.command(name="stopl")
    async def stop_loop(self, ctx):
        if self.set_status.is_running():
            self.set_status.cancel()
        else:
            await ctx.send(embed=cmd_error("Task is already stopped."))
            return
        await ctx.send(embed=info_msg("Stopped updating status"))

    @admin.command(name="startl")
    async def start_loop(self, ctx):
        if self.set_status.is_running():
            await ctx.send(embed=cmd_error("Task is already running."))
            return
        else:
            self.set_status.start()
        await ctx.send(embed=info_msg("Started updating status."))

    @admin.command(name="statusl")
    async def status_loop(self, ctx):
        embed = loop_status(
            self.set_status.is_running(), 
            self.set_status.next_iteration
        )
        await ctx.send(embed=embed)

    @admin.command(name="kill")
    async def kill_bot(self, ctx):
        await ctx.send(f"NOOOOO PLEASE {self.bot.get_emoji(1145147159260450907)}") # :cri: emoji

        def check(reaction, user):
            return self.bot.is_owner(user) and reaction.emoji == self.confim_emote

        try:
            await self.bot.wait_for("reaction_add", timeout=10.0, check=check)
        except TimeoutError:
            await ctx.send(self.bot.get_emoji(994378239675990029))
        else:
            await ctx.send(self.bot.get_emoji(1145090024333918320))
            exit(0)

    @admin.command()
    async def lock(self, ctx):
        if self.locked:
            await ctx.send(embed=cmd_error("Commands already locked."))
        else:
            self.locked = True
            await ctx.send(embed=info_msg("Commands now locked."))

    @admin.command()
    async def unlock(self, ctx):
        if not self.locked:
            await ctx.send(embed=cmd_error("Commands already unlocked."))
        else:
            self.locked = False
            await ctx.send(embed=info_msg("Commands now unlocked."))

    @admin.command()
    async def starboard(self, ctx, channel: TextChannel):
        user_cog = self.bot.get_cog('UserCog')
        user_cog.starboard_id = channel.id  # known type checking error
        await ctx.send(embed=info_msg(f"Set starboard to {channel}"))

    @admin.command()
    async def minstars(self, ctx, minstars: int):
        user_cog = self.bot.get_cog('UserCog')
        user_cog.star_threshold = minstars  # known type cheking error
        await ctx.send(embed=info_msg(f"Set minimum stars to {minstars}"))

    # check if commands are locked
    async def bot_check(self, ctx):
        if await self.bot.is_owner(ctx.author):
            return True
        return not self.locked

    # checks if the author is the owner
    async def cog_check(self, ctx):
        return await self.bot.is_owner(ctx.author)

