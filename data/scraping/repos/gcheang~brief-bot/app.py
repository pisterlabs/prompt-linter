"""
Main module for the BriefBot discord app

Imports
-------
os                  : operating system calls
traceback           : error handling and calls
dotenv              : environment variables
discord             : discord library for development
discord.ext         : discord commands
cohere_functions    : module for cohere functions

Functions:
----------
rundown_helper      : Returns a summary paragraph relating to messages parsed by cohere
on_ready            : Readies the bot to receive commands from users

Bot Commands:
-------------
ping                : Returns a string pong when user gives command
commands            : Returns a list of commands that the user can execute
summarize           : Sends a summary message in dicord containing a summary of messages
                      parsed by cohere
rundown             : Returns a summary paragraph relating to messages parsed by cohere
                      or an error message if given an incorrect input
emotion             : Returns an emotion based on the messages parsed by cohere
move                : Moves the bot to the call that the user is currently in
"""

# imports
import os
from dotenv import load_dotenv

import discord
from discord.ext import commands

import discord.voice_client

from cohere_functions import generate, identify_emotion_v2
from ntranscribe import transcribe


# loading .env
load_dotenv()
TOKEN = os.getenv('BOT_TOKEN')

# setting up discord client
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

connections = {}

async def rundown_helper(ctx, num):
    """
    Returns a summary paragraph relating to messages parsed by cohere

    Parameters
    ----------
    ctx : context
        context of command, contains metadata including message history
    num : int
        number of messages to parse through

    Returns
    -------
    str
        summary of parsed messages
    """
    messages = []
    rundown_input = ""

    # needs to be in chronological order
    async for msg in ctx.channel.history(limit=num+1):
        messages.insert(0, msg)

    # ignore the command call itself
    messages.pop()

    for msg in messages:
        rundown_input += msg.author.name.strip() + ': "' + msg.content.strip() + '"\n'
    print(rundown_input)

    response = generate(rundown_input)
    return response

@bot.event
async def on_ready():
    """
    Readies the bot to receive commands from users
    """
    activity = discord.Game(name="!help", type=3)
    await bot.change_presence(status=discord.Status.online, activity=activity)
    print("Bot is ready!")

# bot commands
@bot.command()
async def ping(ctx):
    """
    Returns a string pong when user gives command
    """
    await ctx.send("Pong!")

bot.remove_command('help')

@bot.command('help')
async def _help(ctx):
    """
    Returns a list of commands that the user can execute
    """
    await ctx.send(
    "**SUPPORTED COMMANDS**\n" +
    "!ping: Play pingpong with BriefBot.\n\n" +
    "!help: Shows commands (this message).\n\n" +
    "!summarize <message>: Summarizes **<message>**. " +
    "Only supports messages of length greater than 250.\n\n" +
    "!rundown <number>: Summarizes the last **<number>** messages in this channel. " +
    "Only supports long conversations.\n\n!emotion <message>: " +
    "Attempts to classify the emotion of **<message>**."
    )

@bot.command()
async def summarize(ctx, *, args=None):
    """
    Sends a summary message in dicord containing a summary of messages parsed
    by cohere

    Parameters
    ----------
    ctx : context
        context of command, includes original message call to command

    Returns
    -------
    str
        summary of parsed messages
    """
    response = generate("".join(args))
    await ctx.send(response)

@bot.command()
async def rundown(ctx, *, args=None):
    """
    Returns a summary paragraph relating to messages parsed by cohere
    or an error message if given an incorrect input

    Parameters
    ----------
    ctx : context
        context of command, contains metadata of command call
    args : list[str]
        list of arguments provided by the user

    Returns
    -------
    str
        summary of parsed messages or error message based on error type
    """
    print("rundown args:", args)
    if args is None:
        print("No args provided")
        response = "Format: !rundown <number>"
    else:
        args_split = args.split(" ")
        if len(args_split) > 1:
            print("Too many args:", len(args_split))
            response = "Format: !rundown <number>"
        else:
            try:
                print("args_split[0]:", args_split[0])
                num = int(args_split[0].strip())
                if num > 100:
                    response = "That's quite a lot of messages! :astonished:\nMax is 100."
                else:
                    response = await rundown_helper(ctx, num)
            except ValueError:
                response = "Format: !rundown <number>"
    await ctx.send(response)

@bot.command()
async def emotion(ctx, *, args=None):
    """
    Returns an emotion based on the messages parsed by cohere

    Parameters
    ----------
    ctx : context
        context of command, contains metadata of command call
    args : list[str]
        list of arguments provided by the user

    Returns
    -------
    str
        an emotion
    """
    response = identify_emotion_v2("".join(args))
    await ctx.send(response)

async def after_record(sink, channel):
    """
    Callback function after recording has finished

    It will disconnect the bot from the voice channel and transcribe the audio
    file for each user in the channel during the recording

    Parameters
    ----------
    sink : discord.sinks.Sink
        sink object that contains the audio data
    channel : discord.VoiceChannel
        voice channel that the bot is connected to
    *args : list
        list of arguments
    
    Returns
    -------
    None
    """

    await sink.vc.disconnect()

    files = []
    transcripts = []
    for user_id, audio in sink.audio_data.items():
        print(audio.file)
        files.append(discord.File(audio.file, f"{user_id}.{sink.encoding}"))

        # create a temporary wav file just so we can transcribe it
        stereo = f"{user_id}-stereo.wav"
        mono = f"{user_id}.wav"

        with open(stereo, "wb") as stereo_file:
            print(stereo_file.name)
            stereo_file.write(audio.file.read())

        # delete mono file if it exists
        if os.path.exists(mono):
            os.remove(mono)

        # convert stereo to mono
        # ffmpeg -i input.wav -ac 1 -y output.wav
        os.system(f"ffmpeg -i {stereo} -ac 1 -y {mono}")

        transcripts.append(f'<@{user_id}>:\n{transcribe(mono)}')
        os.remove(stereo)
        os.remove(mono)

    for transcript in transcripts:
        await channel.send(transcript)

@bot.command()
async def record(ctx):
    """
    Records the audio of users in the voice channel that the user is in

    Parameters
    ----------
    ctx : context

    Returns
    -------
    None
    """

    voice = ctx.author.voice

    if not voice:
        await ctx.send("You are not in a voice channel!")
        return

    voice_channel = await voice.channel.connect()

    connections.update({ctx.guild.id: voice_channel})

    voice_channel.start_recording(
        discord.sinks.WaveSink(),
        after_record,
        ctx.channel
    )

    await ctx.send("Recording...")

@bot.command()
async def stop(ctx):
    """
    Stops the recording of the bot
    Triggers the callback function after_record to transcribe the audio file

    Parameters
    ----------
    ctx : context

    Returns
    -------
    None
    """

    if ctx.guild.id in connections:
        voice_channel = connections[ctx.guild.id]
        voice_channel.stop_recording()

        del connections[ctx.guild.id]

        # await ctx.delete()
    else:
        await ctx.respond("I am currently not recording here.")

# run the bot!
bot.run(TOKEN)
