import discord
import os # default module
from dotenv import load_dotenv
from CohereLayer import *
import asyncio


load_dotenv() # load all the variables from the env file
bot = discord.Bot()

@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")

@bot.slash_command(name = "hello", description = "Say hello to the bot")
async def hello(ctx):
    await ctx.respond("Hey!", ephemeral=True)

@bot.slash_command(name = "say", description = "type what you want bot to say")
async def say(ctx, text: str):
    await ctx.respond(text)

@bot.slash_command(name = "cupid_gen", description = "Bot will help generate a response")
async def cupid_gen(ctx):
    # message = await ctx.fetch_message(ctx.last_message_id)
    await ctx.response.defer()
    message = (await ctx.history(limit=1).flatten())[0].content
    print(message) # TESTING
    response = generate_response(message)
    
    await asyncio.sleep(delay=0)
    await ctx.followup.send(response)
    # await ctx.respond(response)

# **************** Summarizer *******************
@bot.slash_command(name = "cupid_sum", description = "Bot will summarize the previous messages")
async def cupid_sum(ctx):
    await ctx.response.defer()

    message_objects = (await ctx.history(limit=3).flatten())
    message_objects.reverse()
    # Add all messages in the past 3 that were sent by the other person (not command user) to a single string
    # We assume they are apart of the same sentence
    message = ""
    for m in message_objects:
        if m.author != ctx.user:
            message += m.content + " "

    print(message) # TESTING
    response = summarize_text(message)
    # response = generate_response(message)
    await asyncio.sleep(delay=0)
    await ctx.followup.send(response)


# Generate a "specific type of message" given some inputs of the tone of voice etc.
# Ie. Generate a good morning message in happy and sweet tone 



bot.run(os.getenv('TOKEN')) # run the bot with the token