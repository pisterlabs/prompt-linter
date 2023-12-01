import asyncio
import collections
import multiprocessing
import os
import queue
import random

from functools import partial
import collections.abc
import functools
import io
import logging
import os
import discord
import discord.ext
import discord.ext.commands
import openai
from discord.ext import tasks
from discord import TextChannel

# -------------------------------------------------------------------------

OPENAI_API_KEY = ""
gpt356 = "gpt-3.5-turbo-0613"
gpt35616k = "gpt-3.5-turbo-16k-0613"
gpt4 = "gpt-4"
gpt46 = "gpt-4-0613"

bot_token = "" 
with open("bot_token.txt", "r") as f:
    bot_token = f.read().strip()

with open("apikey.txt", 'r') as r:
    OPENAI_API_KEY = r.read().strip()

openai.api_key = OPENAI_API_KEY

def llmagent(instruction_array, model, temperature):
    failed = True
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                messages=instruction_array
            )
            responsestr = response.choices[0].message.content
            failed = False

        except Exception as e:
            print(f"OpenAI API returned an API Error: {e}")
            
        if failed == False:
            print(responsestr)
            return responsestr

# -------------------------------------------------------------------------

BOT_COMMAND_PREFIX = '/'
intents                 = discord.Intents.default()
intents.guilds          = True
intents.dm_messages     = True
intents.dm_reactions    = True
intents.message_content = True
intents.messages        = True
intents.reactions       = True
intents.guild_messages  = True
intents.guild_reactions = True
intents.guild_typing    = True
intents.typing          = True
bot = discord.ext.commands.Bot(command_prefix = BOT_COMMAND_PREFIX, intents=intents)
buffer_log = io.StringIO()
loghandler = logging.StreamHandler(buffer_log)
log        = logging.getLogger('discord')

#------------------------------------------------------------------------

class ContentGenerator:
    def __init__(self, sys_msg, callback=None):
        self.sys_msg = sys_msg
        self.instruction_array = []
        self.callback = callback
        self.generated_text = ""
        self.instruction_array.append({"role": "system", "content": f"{sys_msg}."})

    def add_user_msg(self, user_msg):
        self.instruction_array.append({"role": "user", "content": f"{user_msg}"})

    def run_content_generator(self, model=gpt46, temperature=0):
        self.generated_text = llmagent(self.instruction_array, model, temperature)

    def get_generated_text(self):
        return self.generated_text

    def clear_instruction_array(self):
        self.instruction_array = []
        self.instruction_array.append({"role": "system", "content": f"{self.sys_msg}."})

    def edit_sys_message(self, newsysmsg):
        self.sys_msg = newsysmsg
        self.instruction_array[0] = ""
        self.instruction_array[0] = {"role": "system", "content": f"{newsysmsg}."}

    def run_callback(self):
        self.callback(self)


generator_sysmsg_dict = dict()

generator_sysmsg_dict["profiler"] = "Given a transcript of a conversation, make a participant profile for each individual involved in the conversation. Base the profile on the content of each users interests, concersn, and opinions. Don't include information about personality, just their alignment with the given topics, their strength of feeling, and their level of insight. Return just the profiles, don't offer any additional opinions or thoughts."
generator_sysmsg_dict["summary"] = "Given a transcript of a conversation, list the topics discussed. Under each topic, list the points raised. Under each point raised, name who the person who made the point. Return just the summary, don't offer any additional opinions or thoughts."

generator_sysmsg_dict["convobot"] = "You are given a list of participant profiles for each invidual in a conversation, and a summary of the conversation so far. Reply to queries about the conversation, do not offer your own thoughts or opinions. Any suggestions or additional thoughts in your reply must be aligned with helping clarify the various opinions and positions, with the ultimate goal of facilitating decision making and finding a path toward building a consensus. Reply with just the conversational reply, don't make any presentation remarks around the reply itself."


# -------------------------------------------------------------------------

class GenericButton(discord.ui.Button):
    def __init__(self,
                 style,
                 label,
                 custom_id,
                 callback,
                 timeout = None):
        super().__init__(style=style, label=label, custom_id=custom_id)
        self.callback_func = callback
        self.clickers = set()
        self.id = custom_id

    async def callback(self, interaction: discord.Interaction):
        if self.id == "ask":
            print(self.id)
            if dictionary[session_name]["participants"][interaction.user.name]["replies"] == []:
                await interaction.user.send("You must provide a response, else say 'no reponse', before clicking 'finished'")
                return
            
            if interaction.user.id in self.clickers:
                await interaction.user.send("You have already clicked this button")
                return
        
        if interaction.user.id in self.clickers:
            await interaction.response.send_message("You have already clicked this button", ephemeral=True)
            return

        self.clickers.add(interaction.user.id)
        await self.callback_func(interaction, self.custom_id)

# -------------------------------------------------------------------------

message_transcript = dict()
dictionary = dict()
session_name = ""


def create_session(session):
    global session_name 
    session_name = session
    message_transcript[session_name] = {"transcript": [], "message": []}
    dictionary[session_name] = {"generators": [], "participants": {}, "round_status": "inactive", "round_question": "none"}
    generator_reset()
    generator_constructor("convobot")


def add_participant(User):
    if dictionary[session_name]["round_status"] == "inactive":
        message_transcript[session_name]["transcript"] = []
        message_transcript[session_name]["message"] = []
        dictionary[session_name]["round_status"] = "active"

    if User.name not in dictionary[session_name]["participants"]:
        dictionary[session_name]["participants"][User.name] = {"user": User, "status": "active", "replies": []}

    else:
        dictionary[session_name]["participants"][User.name]["replies"] = []
        dictionary[session_name]["participants"][User.name]["status"] = "active"


def generator_constructor(type):
    if dictionary[session_name]["generators"] is not None:
        dictionary[session_name]["generators"].append(ContentGenerator(generator_sysmsg_dict["profiler"]))
        dictionary[session_name]["generators"].append(ContentGenerator(generator_sysmsg_dict["summary"]))
        dictionary[session_name]["generators"].append(ContentGenerator(generator_sysmsg_dict[type]))
    else:
        print("Error: No session exists")

def generator_summary(model, temperature):
    topic = dictionary[session_name]["round_question"]
    transcript = f"Topic: {topic} " + " ".join(msg for msg in message_transcript[session_name]["transcript"])

    if dictionary[session_name]["generators"] is not None:
        dictionary[session_name]["generators"][1].add_user_msg(transcript)
        dictionary[session_name]["generators"][1].run_content_generator(model, temperature)


def generator_profiler(model, temperature):
    
    transcript = "Transcript: " + " ".join(msg for msg in message_transcript[session_name]["transcript"])
    if dictionary[session_name]["generators"] is not None:
        dictionary[session_name]["generators"][0].add_user_msg(transcript)
        dictionary[session_name]["generators"][0].run_content_generator(model, temperature)


def generator_convobot(model, temperature):
    generator_profiler(model, temperature)
    dictionary[session_name]["generators"][2].add_user_msg("Participant Profiles: " + dictionary[session_name]["generators"][0].get_generated_text() + "  Discussion summary: " +   dictionary[session_name]["generators"][1].get_generated_text())


def generator_reset():
    if dictionary[session_name]["generators"] is not None:
        for generator in dictionary[session_name]["generators"]:
            generator.clear_instruction_array()
            generator.generated_text = ""
    else:
        print("Error: No Generators exist")
        return


async def send_summary():
    generator_summary(gpt46, 0)
    for participant in dictionary[session_name]["participants"]:
        await dictionary[session_name]["participants"][participant]["user"].send(f"Round over! \n\n{dictionary[session_name]['generators'][1].get_generated_text()}")



# -------------------------------------------------------------------------
# Bot events.

# Initial Event
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')



# On Message Event
@bot.event
async def on_message(message):

    if not session_name:
        await bot.process_commands(message)
        return

    if message.content.startswith("/delib"):
        await bot.process_commands(message)
        message_transcript[session_name]["transcript"].append(f"{message.content[:12]}")
        message_transcript[session_name]["transcript"]
        return
    

    if dictionary is not None:
        if dictionary[session_name]["round_status"] == "active":

            for participant in dictionary[session_name]["participants"]:
                if message.channel == dictionary[session_name]["participants"][participant]["user"].dm_channel:
                    if message.author != bot.user:
                        print("message from participant")
                        message_transcript[session_name]["transcript"].append(f"{message.author.name}: {message.content}")
                        message_transcript[session_name]["message"].append(message)
                        dictionary[session_name]["participants"][participant]["replies"].append(message)
                        print(message_transcript[session_name]["transcript"][-1])
            return

    if isinstance(message.channel, discord.DMChannel):   
        if message_transcript[session_name]["transcript"] is not None:
            if message.author != bot.user:
                message_transcript[session_name]["transcript"].append(f"{message.author.name}: {message.content}")
                message_transcript[session_name]["message"].append(message)



    if message.reference is not None:
        print("getting referenced message")
        referenced_message = message.reference.resolved
        if referenced_message.author == bot.user:
            await runconvo(message)
        return

    await bot.process_commands(message)

# -------------------------------------------------------------------------
# Reply to bot methods: (called in on_message)

# Accord reply

async def runconvo(message):

    generator = dictionary[session_name]["generators"][2]
    if generator.generated_text == "":
        generator_convobot(gpt46, 0)

    generator.add_user_msg(f"{message.author.name}: {message.content}")
    generator.run_content_generator(gpt46, 0)
    _reply = generator.get_generated_text()
    await message.reply(_reply)

# -------------------------------------------------------------------------
# Button Callback Methods:

async def button_action(interaction, custom_id):
    '''
    This method is called when the user clicks the button.
    
    '''
    question = dictionary[session_name]["round_question"]

    clicker = interaction.user.name

    if custom_id == "deliberation":
        add_participant(interaction.user)
        button = GenericButton(style=discord.ButtonStyle.green, label='Finished', custom_id='ask', callback=button_action)
        view = discord.ui.View()
        view.add_item(button)
        await interaction.user.send(f"Question: {question}", view=view)

        for participant in dictionary[session_name]["participants"]:
            if participant == clicker:
                continue
            await dictionary[session_name]["participants"][participant]["user"].send(f"{clicker} has joined the deliberation")
        await interaction.response.send_message(f"{clicker} has joined the deliberation", ephemeral=False)
    
    if custom_id == "ask":
        if dictionary[session_name]["participants"][clicker]["replies"] is None:
            return
        dictionary[session_name]["participants"][interaction.user.name]["status"] = "finished"
        
        for participant in dictionary[session_name]["participants"]:
            if dictionary[session_name]["participants"][participant]["status"] == "active":
                return
        dictionary[session_name]["round_status"] = "inactive"
        await send_summary()



#--------------------------------------------------------------------------
# Session commands

@bot.command(name="start")
async def start_command(ctx, *, session_id):
    session_name = session_id
    create_session(session_name)
    await ctx.send(f"Session {session_name} created")



@bot.command(name="delib")
async def deliberation_command(ctx, *, question):
    dictionary[session_name]["round_question"] = question
    button = GenericButton(style=discord.ButtonStyle.green, label='Join Deliberation', custom_id="deliberation", callback=button_action)
    view = discord.ui.View()
    view.add_item(button)
    await ctx.send(f'{question}', view=view)



@bot.command(name="query")  
async def convo_roll(ctx):
    global dictionary
    try:
        for participant in dictionary[session_name]["participants"]:
            await dictionary[session_name]["participants"][participant]["user"].send("Hello everyone, Reply to this message if you want to get detailed information about the session. Replying to any follow up messages will get me to respond.")

    except Exception as e:
        print(f"Error occurred: {e}")



@bot.command(name="altconvo")
async def alter_convo(ctx, *, new_system_message):
    dictionary[session_name]["generators"][2].edit_sys_message(f"""{new_system_message}""")




# -------------------------------------------------------------------------
# Utility stuff

@bot.command(name="cmdhelp")
async def help(ctx):
    help_menu = """# -------------------------------------------
    # Session commands:

    /start <session_name> - prints this menu.
    /delib <question> - starts a deliberation with the question.
    /altconvo, <newsystemmessage>, changes the system message for the session summary reply-bot
    /query - sends a message to all participants to query the bot for information about the session.

 
    # -------------------------------------------

    # Admin commands:

    /cmdhelp - prints this menu.

    /scrape, channel, scope - scrapes the channel for messages within the scope.
    /shutdown - shuts down the bot.
    /censor <channel> <user> <scope> - censors the user's messages in the channel within the scope.
    /purge <channel> - purges all messages within the channel.
    /purgecat <category> - purges all messages in each channel within the channel category.
    
    """
    print(help_menu)



@bot.command(name="scrape")
async def collect_messages(ctx, channel: discord.TextChannel, scope: int):
    '''
    Fill up the message array with the chosen channel's messages within the scope. 
    Be careful not to have a huge scope because it can fill up the context window for the summary generators.

    '''
    await ctx.send(f"Scope is set to maximum {scope}.")
    async for message in channel.history(limit=scope):  # adjust the limit as needed
        try:
            if session_name not in message_transcript:
                message_transcript[session_name] = {"transcript": [], "message": []}

            print(f"{message.author.name}: {message.content}")
            message_transcript[session_name]["transcript"].append(f"{message.author.name}: {message.content}")
            message_transcript[session_name]["message"].append(message)
            #await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Error occurred: {e}")

# -------------------------------------------------------------------------
# Admin commands

@bot.command(name="shutdown")
async def shutdown(ctx):
    await ctx.send("Shutting down...")
    await bot.close()


@bot.command(name="censor")
async def delete(ctx, channel: discord.TextChannel, user: discord.Member, num: int):
    def is_user(m):
        return m.author == user
    deleted = await channel.purge(limit=num, check=is_user)
    await ctx.send(f'Deleted {len(deleted)} message(s)')


@bot.command(name="purge")
async def purge(ctx, channel: discord.TextChannel):
    await channel.purge()


@bot.command(name="purgecat")
async def purge(ctx, category: discord.CategoryChannel):
    for channel in category.text_channels:
        await channel.purge()


# -------------------------------------------------------------------------
bot.run(bot_token)
