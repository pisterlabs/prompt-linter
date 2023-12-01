import discord
import os

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_functions import langchain_helper
from discord_functions import discord_app

embeddings = OpenAIEmbeddings()

openai_api_key  = os.getenv('OPENAI_API_KEY')
discord_token = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True

client = discord_app.get_discord_client(intents, 'gpt-3.5-turbo', 0.2)

@client.event
async def on_ready():
    print(f'Bot has logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if client.user.mentioned_in(message):
        async with message.channel.typing():
            try:
                response = langchain_helper.get_qa_from_query(client, message.content)
                await message.channel.send(response)
            except Exception as e:
                print(f"An exception occurred: {str(e)}")
                await message.channel.send('I got an error while processing your query')
    await client.process_commands(message)

@client.command()
async def set_video(ctx, link):
    async with ctx.channel.typing():
        try:
            db, video_meta = langchain_helper.set_video_as_vector(link, embeddings)
            client.video_db = db
            client.video_meta = video_meta
            client.qa = langchain_helper.get_qa(client)
            await ctx.channel.send('I checked the video!')
        except Exception as e:
            print(f"An exception occurred: {str(e)}")
            await ctx.channel.send('I got an error while watching the video')

client.run(discord_token)
