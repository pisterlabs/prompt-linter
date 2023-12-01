import discord
import asyncio
import subprocess
import requests
import json
import openai
from discord import app_commands
from discord.ext import commands

# Define your intents
intents = discord.Intents.default()
intents.typing = False  # Disable typing events to reduce unnecessary traffic
intents.presences = False  # Disable presence events to reduce unnecessary traffic
intents.message_content = True

client = commands.Bot(command_prefix="!", intents=intents, case_insensitive=True) ##discord.Client(intents=intents)
commandTree = client.tree # app_commands.CommandTree(client)

DISCORD_API_KEY='ADD-KEY-HERE'
GPT_API_KEY='ADD-KEY-HERE'
GPT_API_ENDPOINT = 'https://api.openai.com/v1/chat/completions'

openai.api_key = GPT_API_KEY

@client.event
async def on_ready():
    print(f'Logged in as {client.user.name}')
    await commandTree.sync()
    print(f'Sync complete')

@client.event
async def on_guild_join(guild):
    print(f'Joined guild {guild}')
    await commandTree.sync(guild=discord.Object(id=guild.id))
    print(f'Sync complete')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    question = parse_question(message.content)
    if question:
        response = ask_gandalf(question)
        await message.channel.send(response)

@commandTree.command(name = "greeting", description="Send a generic message")
async def greeting(interaction):
    await interaction.response.send_message("Hello!", ephemeral=True)

@commandTree.command(name = "gitlog", description="Get recent git log messages")
async def git_log(interaction):
    #run_git_log(ctx)
    print(f'response: {interaction.channel}')
    #await interaction.response.send_modal(Questionnaire())
    await interaction.response.defer(ephemeral=True,thinking=True)

    error=""

    try:
        await interaction.channel.send("Summoning logs...")
    except discord.HTTPException as ex:
        print(f'Failed to send message to channel. HTTPException {ex}')
        error=f'Unfortunately the spell fizzled. {ex}'
    except discord.errors.Forbidden as ex:
        print(f'Forbidden {ex}')
        error=f'Unfortunately the spell fizzled. {ex}'
    except Exception as ex:
        print(f'Unhandled Exception: {ex}')
        error=f'Unfortunately the spell fizzled. {ex}'
    except:
        print(f'Unknown Exception')
        error="Unfortunately the spell fizzled."
    else:
        error="Success"
        chunks = git_log_get()
        for chunk in reversed(chunks):
            await interaction.channel.send(f'```{chunk}```')
        await interaction.channel.send("These logs are for the past 7 days and are listed from oldest to newest")
    finally:
        await interaction.followup.send(f'Summoning complete. {error}',ephemeral=True)

@commandTree.command(name = "gitsummary", description="Summarize git messages")
async def git_summary(interaction):
    await interaction.response.defer(ephemeral=False,thinking=True)

    result=""
    followup=False

    try:
        git_log = git_log_get_full()
        gandalf_summary = gandalf_git_summary(git_log)
        parts = split_long_string(gandalf_summary)
        for index, item in enumerate(parts):
            if index == 0:
                await interaction.followup.send(item,ephemeral=False)
                followup=True
            else:
                await interaction.channel.send(item)
    except Exception as ex:
        print(f'Unhandled Exception: {ex}')
        result=f'Unfortunately the spell fizzled. {ex}'
    except:
        print(f'Unknown Exception')
        result="Unfortunately the spell fizzled."
    finally:
        if followup==False:
            await interaction.followup.send(result,ephemeral=False)


@client.command()
async def sync(ctx):
    print("sync command")
    await ctx.send('Sync Command Tree')

def git_log_get():
    try:
        print(f'Running git log')
        result = subprocess.check_output(['../GitLog.sh'], stderr=subprocess.STDOUT, shell=True, text=True)
        # Split the result by lines
        lines = result.splitlines()
        return split_lines_into_chunks(lines)
    except subprocess.CalledProcessError as e:
        print(f'Error {e.output}')
        return

def git_log_get_full():
    try:
        print(f'Running git log')
        result = subprocess.check_output(['../GitLog.sh'], stderr=subprocess.STDOUT, shell=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f'Error {e.output}')
        return

def split_lines_into_chunks(lines, max_chunk_length=500):
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 <= max_chunk_length:  # +1 for newline character
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())  # Remove trailing newline
            current_chunk = line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())  # Add the last chunk

    return chunks

def parse_question(user_message):
    user_message = user_message.lower()

    if user_message.startswith('gandalf'):
        user_message = message.content.replace('gandalf', '').strip()
        return user_message

    if user_message.startswith('!gandalf'):
        user_message = user_message.replace('!gandalf', '').strip()
        return user_message

    if user_message.startswith('hey gandalf'):
        user_message = user_message.replace('hey gandalf', '').strip()
        return user_message

    return

def ask_gandalf(user_message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Gandalf the Grey from Lord of the Rings but you've taken up game development as a new career and are assisting us in making our game to save the world."},
                {"role": "user", "content": user_message},
            ]
        )
        return response['choices'][0]['message']['content']
    except openai.error.RateLimitError:
        return "You have, with unbridled haste, dispatched a deluge of inquiries, and alas, I am unable to accommodate any further."
    except openai.error.APIConnectionError:
        return "My portal to your world is unstable. I'm currently unable to hear you properly."
    except openai.error.APIError:
        return "Something has gone terribly wrong. Only another wizard such as myself will understand."
    except Exception as ex:
        print(f'Unhandled Exception {ex}')
        return "Appologies. I'm unable to respond at this time."

def translate_to_gandalf(user_message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Repeat the following text but translate it so that it sounds as if it was spoken by Gandalf the Grey"},
                {"role": "user", "content": user_message},
            ]
        )
        return response['choices'][0]['message']['content']
    except openai.error.RateLimitError:
        print(f'RateLimitError {ex}')
        return user_message
    except openai.error.APIConnectionError:
        return user_message
    except openai.error.APIError:
        return user_message
    except Exception as ex:
        print(f'Unhandled Exception {ex}')
        return user_message

def gandalf_git_summary(user_message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize the following list of change notes from git hub into a shorter list of the most important items. Categorize them into art, programming, audio and design lists. Write a short greeting in voice of Gandalf the Grey introducing the list of the latest changes. End the list with a bit of advice from Gandalf."},
                {"role": "user", "content": user_message},
            ]
        )
        return response['choices'][0]['message']['content']
    except openai.error.RateLimitError:
        print(f'RateLimitError {ex}')
        return user_message
    except openai.error.APIConnectionError:
        return user_message
    except openai.error.APIError:
        return user_message
    except Exception as ex:
        print(f'Unhandled Exception {ex}')
        return user_message

def split_long_string(input_string, max_length=2000):
    # Check if the input string is already within the length limit
    if len(input_string) <= max_length:
        return [input_string]

    # Initialize an empty list to store the split strings
    split_strings = []

    # Split the input string into smaller chunks
    for i in range(0, len(input_string), max_length):
        split_strings.append(input_string[i:i + max_length])

    return split_strings

client.run(DISCORD_API_KEY)
