"""
RinBot v1.9.1 (GitHub release)
made by rin
"""

# Make sure cache dirs exist
import os
folders = ["cache", "cache/fun", "cache/chatlog", 
           "cache/favorites", "cache/histories", 
           "cache/stablediffusion", "cache/ocr"]
for folder in folders:
    if not os.path.exists(f"{os.path.realpath(os.path.dirname(__file__))}/{folder}"):
        os.makedirs(f"{os.path.realpath(os.path.dirname(__file__))}/{folder}")
        print(f"[init.py]-[Info]: Created directory '{folder}'")

# Imports
import subprocess, asyncio, platform, sys, aiosqlite, exceptions, discord, time, datetime
from discord.ext import commands, tasks
from discord.ext.commands import Bot, Context
from program.logger import logger
from program import db_manager
from langchain.llms.koboldai import KoboldApiLLM
from dotenv import load_dotenv
from program.helpers import strtobool

# Load env
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
BOT_PREFIX = os.getenv('BOT_PREFIX')
INIT_WELCOME_CHANNEL_ID = os.getenv('INIT_WELCOME_CHANNEL_ID')
RULE_34_ENABLED = strtobool(os.getenv("RULE_34_ENABLED"))
BOORU_ENABLED = strtobool(os.getenv('BOORU_ENABLED'))
AI_ENABLED = strtobool(os.getenv('AI_ENABLED'))
AI_CHAR_NAME = os.getenv('AI_CHAR_NAME')
AI_ENDPOINT_KOBOLD = os.getenv('AI_ENDPOINT_KOBOLD')
AI_CHANNEL = os.getenv('AI_CHANNEL')
AI_CHAT_HISTORY_LINE_LIMIT = os.getenv('AI_CHAT_HISTORY_LINE_LIMIT')
AI_MAX_NEW_TOKENS = os.getenv('AI_MAX_NEW_TOKENS')
AI_LANGUAGE = os.getenv('AI_LANGUAGE')

# Check presence of ffmpeg
if platform.system() == 'Windows':
    if not os.path.isfile(f"{os.path.realpath(os.path.dirname(__file__))}/bin/ffmpeg.exe"):
        sys.exit("[init.py]-[Error]: 'ffmpeg.exe' not found.")
elif platform.system() == 'Linux':
    try:
        # Try to execute ffmpeg if on linux
        result = subprocess.run(['ffmpeg', '-version'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
        if result.returncode != 0:
            sys.exit("[init.py]-[Error]: 'ffmpeg' not found on this system, please install it, or if it is installed, check if it is available in PATH.")
    except FileNotFoundError:
        sys.exit("[init.py]-[Error]: 'ffmpeg' not found on this system, please install it, or if it is installed, check if it is available in PATH.")

# Bot intentions (many)
intents = discord.Intents.all()
intents.dm_messages = True
intents.dm_reactions = True
intents.dm_typing = True
intents.emojis = True
intents.guild_messages = True
intents.guild_reactions = True
intents.guild_scheduled_events = True
intents.guild_typing = True
intents.guilds = True
intents.integrations = True
intents.invites = True
intents.voice_states = True
intents.webhooks = True
intents.members = True
intents.message_content = True
intents.moderation = True
intents.presences = True
intents.emojis_and_stickers = True
intents.messages = True
intents.reactions = True
intents.typing = True
intents.bans = True

# Bot
bot = Bot(
    command_prefix=commands.when_mentioned_or(BOT_PREFIX),
    intents=intents,
    help_command=None,)
bot.logger = logger

# Vars
freshstart = True
message_count = {}
time_window_milliseconds = 5000
max_msg_per_window = 5
author_msg_times = {}

# Will I use AI?
if AI_ENABLED:
    # Specific AI settings
    bot.endpoint = str(AI_ENDPOINT_KOBOLD)
    if len(bot.endpoint.split("/api")) > 0:
        bot.endpoint = bot.endpoint.split("/api")[0]
    bot.chatlog_dir = "cache/chatlog"
    bot.endpoint_connected = False
    bot.channel_id = AI_CHANNEL
    bot.num_lines_to_keep = int(AI_CHAT_HISTORY_LINE_LIMIT)
    bot.guild_ids = [int(x) for x in AI_CHANNEL.split(",")]
    bot.debug = True
    bot.char_name = AI_CHAR_NAME
    characters_folder = "ai/Characters"
    cards_folder = "ai/Cards"
    characters = {}
    bot.endpoint_type = "Kobold"
    bot.llm = KoboldApiLLM(endpoint=bot.endpoint, max_length=AI_MAX_NEW_TOKENS)

# Start SQL database
async def init_db():
    async with aiosqlite.connect(
        f"{os.path.realpath(os.path.dirname(__file__))}/database/database.db"
    ) as db:
        with open(
            f"{os.path.realpath(os.path.dirname(__file__))}/database/schema.sql"
        ) as file:
            await db.executescript(file.read())

# Checks if the bot has someone on the 'owners' class
async def check_owners():
    owners = await db_manager.get_owners()
    if len(owners) == 0:
        owner_valid = False
        bot.logger.warning('Fresh database, first start detected, Welcome!')
        bot.logger.info('Please copy and paste your discord user ID so we can add you as a bot owner.')
        while not owner_valid:
            fresh_owner_id:str = input('Your ID: ')
            owner_valid = fresh_owner_id.isnumeric()
            if owner_valid:
                await db_manager.add_user_to_owners(fresh_owner_id)
                await db_manager.add_user_to_admins(fresh_owner_id)
                bot.logger.info("Thanks! You've been added to the 'owners' class, have fun! :D")
            else:
                bot.logger.error('Invalid ID. Make sure it only contains numbers.')
    else:
        bot.logger.info('Owners class filled, proceeding.')

# When ready
@bot.event
async def on_ready() -> None:
    # Initial logger info (splash)
    bot.logger.info("--------------------------------------")
    bot.logger.info(" >   RinBot v1.9.1 (GitHub release)   ")
    bot.logger.info("--------------------------------------")
    bot.logger.info(f" > Logged as {bot.user.name}")
    bot.logger.info(f" > API Version: {discord.__version__}")
    bot.logger.info(f" > Python Version: {platform.python_version()}")
    bot.logger.info(f" > Running on: {platform.system()}-{platform.release()} ({os.name})")
    bot.logger.info("--------------------------------------")
    
    # Check if all members are present in the economy database
    bot.logger.info("Checking economy presence")
    for guild in bot.guilds:
        for member in guild.members:
            await db_manager.add_user_to_currency(member.id, guild.id)
    
    # Load AI channel
    if AI_ENABLED:
        bot.logger.info('Using AI...')
        for items in bot.guild_ids:
            try:
                channel = bot.get_channel(int(items))
                guild = channel.guild
                if isinstance(channel, discord.TextChannel):
                    channel_name = channel.name
                    bot.logger.info(f'AI Text channel: {guild.name} | {channel_name}')
                else:
                    bot.logger.error(f'AI Text channel {bot.channel_id} is not a valid text channel, check your ID or channel settings.')
            except AttributeError:
                bot.logger.error(
                    "Couldn't validate AI text channel, verify the ID, channel settings, and if I have the necessary permissions.")
    
    # Default status
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening, name='to your commands :3'))
        
    # Sync commands with discord
    bot.logger.info("Synching commands")
    await bot.tree.sync()

# Member welcome
@bot.event
async def on_member_join(member:discord.Member):
    # Add new members to the economy database
    await db_manager.add_user_to_currency(int(member.id), int(member.guild.id))
    
    if INIT_WELCOME_CHANNEL_ID.isnumeric():
        channel = bot.get_channel(int(INIT_WELCOME_CHANNEL_ID))
        if channel:
            embed = discord.Embed(
                title=' :star:  New member!',
                description=f'Welcome to {member.guild.name}, {member.mention}!',
                color=0xf5be0a)
            try:
                embed.set_thumbnail(url=member.avatar.url)
            except AttributeError:
                pass
            await channel.send(embed=embed)

# Save new guild ID's when joining
@bot.event
async def on_guild_join(guild):
    await db_manager.add_joined_on(str(guild.id))
    bot.logger.info(f'Joined guild ID: {guild.id}')
    
    # Add all members to the economy database
    user_ids = [i.id for i in guild.members]
    for id in user_ids:
        await db_manager.add_user_to_currency(int(id), int(guild.id))

# Processes standard message commands
@bot.event
async def on_message(message: discord.Message) -> None:
    try:
        if message.author == bot.user or message.author.bot:
            return
        
        # Anti-spam measure
        global author_msg_times
        aid = message.author.id
        ct = datetime.datetime.now().timestamp() * 1000
        if not author_msg_times.get(aid, False):
            author_msg_times[aid] = []
        author_msg_times[aid].append(ct)
        et = ct - time_window_milliseconds
        em = [mt for mt in author_msg_times[aid] 
            if mt < et]
        for mt in em:
            author_msg_times[aid].remove(mt)
        if len(author_msg_times[aid]) > max_msg_per_window:
            await db_manager.update_message_count(message.author.id, message.guild.id, True)
        else:
            await db_manager.update_message_count(message.author.id, message.guild.id)
        
        await bot.process_commands(message)
    except AttributeError:
        pass

# Show executed commands on the log
@bot.event
async def on_command_completion(context: Context) -> None:
    full_command_name = context.command.qualified_name
    split = full_command_name.split(" ")
    executed_command = str(split[0])
    if context.guild is not None:
        bot.logger.info(
            f"Command {executed_command} executed in {context.guild.name} (ID: {context.guild.id}) by {context.author} (ID: {context.author.id})")
    else:
        bot.logger.info(
            f"Command {executed_command} executed by {context.author} (ID: {context.author.id}) on my DMs.")

# What to do when commands go no-no
@bot.event
async def on_command_error(context: Context, error) -> None:
    if isinstance(error, commands.CommandOnCooldown):
        minutes, seconds = divmod(error.retry_after, 60)
        hours, minutes = divmod(minutes, 60)
        hours = hours % 24
        embed = discord.Embed(
            description=f"**Please wait! >-< ** - You can use this command again in {f'{round(hours)} hours' if round(hours) > 0 else ''} {f'{round(minutes)} minutes' if round(minutes) > 0 else ''} {f'{round(seconds)} seconds' if round(seconds) > 0 else ''}.",
            color=0xE02B2B,)
        await context.send(embed=embed)
    elif isinstance(error, exceptions.UserBlacklisted):
        embed = discord.Embed(
            description="You are blocked from using RinBot!", color=0xE02B2B)
        await context.send(embed=embed)
        if context.guild:
            bot.logger.warning(
                f"{context.author} (ID: {context.author.id}) tried running a command on guild {context.guild.name} (ID: {context.guild.id}), but they're blocked from using RinBot.")
        else:
            bot.logger.warning(
                f"{context.author} (ID: {context.author.id}) tried running a command on my DMs, but they're blocked from using RinBot.")
    elif isinstance(error, exceptions.UserNotOwner):
        embed = discord.Embed(
            description="You are not on the RinBot `owners` class, kinda SUS!", color=0xE02B2B)
        await context.send(embed=embed)
        if context.guild:
            bot.logger.warning(
                f"{context.author} (ID: {context.author.id}) tried running a command of class `owner` {context.guild.name} (ID: {context.guild.id}), but they're not a part of this class")
        else:
            bot.logger.warning(
                f"{context.author} (ID: {context.author.id}) tried running a command of class `owner` on my DMs, but they're not a part of this class")
    elif isinstance(error, exceptions.UserNotAdmin):
        embed = discord.Embed(
            description="You are not on the RinBot `admins` class, kinda SUS!", color=0xE02B2B)
        await context.send(embed=embed)
        if context.guild:
            bot.logger.warning(
                f"{context.author} (ID: {context.author.id}) tried running a command of class `admin` {context.guild.name} (ID: {context.guild.id}), but they're not a part of this class")
        else:
            bot.logger.warning(
                f"{context.author} (ID: {context.author.id}) tried running a command of class `admin` on my DMs, but they're not a part of this class")
    elif isinstance(error, commands.MissingPermissions):
        embed = discord.Embed(
            description="You don't have `"
            + ", ".join(error.missing_permissions)
            + "` permissions, which are necessary to run this command!",
            color=0xE02B2B,)
        await context.send(embed=embed)
    elif isinstance(error, commands.BotMissingPermissions):
        embed = discord.Embed(
            description="I don't have `"
            + ", ".join(error.missing_permissions)
            + "` permissions, which are necessary to run this command!",
            color=0xE02B2B,)
        await context.send(embed=embed)
    elif isinstance(error, commands.MissingRequiredArgument):
        embed = discord.Embed(
            title="Error!",
            description=str(error).capitalize(),
            color=0xE02B2B,)
        await context.send(embed=embed)
    else:
        raise error

# Loads extensions (command cogs)
async def load_extensions() -> None:
    ai_ext = ["imagecaption", "languagemodel", "message_handler", "stablediffusion"]
    booru_ext = ["danbooru"]
    e621_ext = ["e621"]
    rule34_ext = ["rule34"]
    sum = ai_ext + booru_ext + e621_ext + rule34_ext
    for file in os.listdir(f"{os.path.realpath(os.path.dirname(__file__))}/extensions"):
        if file.endswith(".py"):
            extension = file[:-3]
            if AI_ENABLED and extension in ai_ext:
                await load_extension(extension)
            elif BOORU_ENABLED and extension in booru_ext:
                await load_extension(extension)
            elif RULE_34_ENABLED and extension in rule34_ext:
                await load_extension(extension)
            is_general = all(extension not in sl for sl in sum)
            if is_general:
                await load_extension(extension)

# Loads an extension
async def load_extension(extension) -> None:
    try:
        await bot.load_extension(f"extensions.{extension}")
        bot.logger.info(f'Extension "{extension}" loaded')
    except Exception as e:
        exception = f"{type(e).__name__}: {e}"
        bot.logger.info(f'Error loading extension "{extension}"\n{exception}')

# Wait 5 seconds when coming from a reset
try:
    if sys.argv[1] == 'reset':
        print('Coming from a reset, waiting for previous instance to finish...')
        time.sleep(5)
except:
    pass

# RUN
asyncio.run(init_db())
asyncio.run(check_owners())
asyncio.run(load_extensions())
bot.run(BOT_TOKEN)