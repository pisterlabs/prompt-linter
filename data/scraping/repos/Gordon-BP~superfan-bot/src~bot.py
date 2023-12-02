import discord
from discord.ext import commands
import cohere
from createData import getDataAsSoup, cleanData, fast_chonk, create_index, create_index
import os
import sys
import pinecone
from pathlib import Path
from app import get_embedding, query_index, prompt_completion
import logging
import pandas as pd
import re
from transformers import GPT2TokenizerFast

# Ensure we have the discord token passed when the Docker image is started
if(os.environ['DISCORD_TOKEN']):
    token = os.environ["DISCORD_TOKEN"]
else:
    raise EnvironmentError("Please set DISCORD_TOKEN env when running your docker image")
    
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents)

# Friendly startup message
@bot.event
async def on_ready() -> None:
    """
    Initializes the bot, global variables, and sets up the logger
    """
    global co
    global co_status
    global pinecone_status
    co_status = False
    pinecone_status = False
    # Let's set up logging!
    discord.utils.setup_logging(level=logging.INFO, root=False)
    global botLogger
    botLogger = logging.getLogger(f"{bot.user}")
    c_handler = logging.StreamHandler()
    c_formatter = logging.Formatter('%(asctime)s %(levelname)s\t%(name)s\t%(message)s')
    c_handler.setFormatter(c_formatter)
    botLogger.addHandler(c_handler)
    botLogger.setLevel(logging.INFO) # <-- THIS!

    # Next let's initialize cohere!
    if(os.environ['COHERE_API_KEY'] != ""):
        try:
            co = cohere.Client(os.environ['COHERE_API_KEY'])
            if(co.check_api_key()['valid']):
                co_status = True
        except:
            co_status = False
    if((os.environ['PINECONE_API_KEY'] != "") and 
        (os.environ['PINECONE_ENV'] != "")):
        try:
            pinecone.init(api_key = os.environ['PINECONE_API_KEY'], host = os.environ['PINECONE_ENV'])
            pinecone_status = True
        except:
            pinecone_status = False
    botLogger.info(f'The bot is ready!')

# Command just to test if the bot is onling
@bot.command()
async def status_check(ctx: discord.ext.commands.context.Context) -> None:
    """
    Checks the presence and validity of the Cohere and Pinecone API keys. 
    """
    msg = "API Key validity status:"
    if(co_status):
        msg += "\n✅ Valid Cohere API Key"
    else: msg += "\n❌ No or invalid Cohere API Key"
    if(pinecone_status):
        msg += "\n✅ Valid Pinecone API Key"
    else: msg += "\n❌ No or invalid Pinecone API Key"
    await ctx.send(msg)

@bot.command()
async def set_cohere_key(ctx: discord.ext.commands.context.Context, key:str) -> None:
    """
    Method to set cohere key
    """
    global co_status
    try:
        os.environ['COHERE_API_KEY'] = str(key)
        co = cohere.Client(os.environ['COHERE_API_KEY'])
        if(co.check_api_key()['valid']):
            co_status = True
        if(co_status):
            await ctx.send("✅ Cohere API key successfully set")
    except:
        await ctx.senc("❌ invalid API key, Cohere key not set ❌ ")
        os.environ['COHERE_API_KEY'] = ""

@bot.command()
async def set_pinecone_key(ctx: discord.ext.commands.context.Context, key:str) -> None:
    """
    Method to set pinecone key
    """
    global pinecone_status
    os.environ['PINECONE_API_KEY'] = str(key)
    try:
        pinecone.init(api_key = os.environ['PINECONE_API_KEY'], host = os.environ['PINECONE_ENV'])
        pinecone_status = True
    except:
        pinecone_status = False
    if(pinecone_status):
        await ctx.send("✅ Pinecone API key successfully set")
    else:
        await ctx.senc("❌ Invalid API key, Pinecone key not set ❌ ")
        os.environ['PINECONE_API_KEY'] = ""


@bot.command()
async def create_dataset(ctx: discord.ext.commands.context.Context, url:str, index:str, **kwargs) -> None:
    """
    Takes a datasource URL and creates a Pinecone Index vector database out of it!

    Parameters:
        ctx: the inherent Discord bot context
        url(str): The data source URL, typically an amazon s3 instance
        index(str): The label for the index to create
        **kwargs: None yet, might add some late like:
            #TODO: add configurable minimum and maximum token length for data chunks
            #TODO: add configurable similarity metrics
            #TODO: optional index overrides
    Returns:
        Nothing, the bot just says a bunch of messages
    """
    # First, check for valid API keys
    if(not(co_status)):
        await ctx.send("❌ Invalid cohere API key, cannot make embeddings ❌")
        await ctx.send("Please use $set_cohere_key to set your API key")
        botLogger.error(f"Cohere API key not set")
        return
    if(not(pinecone_status)):
        await ctx.send("❌ Invalid pinecone API key, cannot make dataset ❌")
        await ctx.send("Please use $set_pinecone_key to set your API key")
        botLogger.error(f"Pinecone API key not set")
        return
    # First, let's type check all of our arguments!
    await ctx.send(f'Creating an index for {url}...')
    if not(isinstance(url, str)):
        await ctx.send(f"The URL is {type(url)} not str")
        botLogger.error(f"Invalid Parameter: The URL is {type(url)} not str")
        return
    elif not(isinstance(index, str)):
        await ctx.send(f"The Index label is type {type(index)} and not a valid string")
        botLogger.error(f"Invalid Parameter: The Index label is {type(url)} not str")
        return
    else:
        botLogger.info("Creating index, please wait")
        #TODO: this is where we unpack and verify the kwargs
        index = await create_index(url, index, maxTokens=128, minTokens=10)
        await ctx.send('Index created! Here are the stats:')
        stats = index.describe_index_stats()
        await ctx.send(str(stats))
        await ctx.send(f"You can now query this index using $query {index} your-question-here")
    
@bot.command()
async def query(ctx:discord.ext.commands.context.Context, index:str, *args) -> None:
    """
    Main method for querying the pinecone index. In discord, it should look like:
    $query my_index what is your name?
    
    Parameters:
        ctx: the inherent Discord context
        index: The name of the pinecone instance you want to query. Might take this out later idk
        *args: The query you want to ask, spaces and all.
    """
    query = ' '.join(args)
    results = await query_index([query], index)
    results_list = [match['metadata']['text'] for match in results['matches']]
    botLogger.info(f"The restults are in! Top results are:\n{results_list}")
    bot_answer = await prompt_completion(query, results_list)
    botLogger.info(f"The final answer is:\n{bot_answer}")
    await ctx.send(f"Here's what I think the answer to '{query}' is:\n{bot_answer}")

bot.run(token)