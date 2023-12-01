import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from openai import OpenAI
import os
import logging

# Load environment variables
load_dotenv()

# be sure to create a .env file with your DISCORD_BOT_TOKEN and OPENAI_API_KEY
# instructions for generating API keys is linked in the description
discord_token = os.getenv("DISCORD_BOT_TOKEN")

# Initialize the OpenAI client outside of the process_chat function to avoid re-initialization on each call
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize log
logging.basicConfig(filename='debug.log', level=logging.INFO)

# Initialize bot context
bot_context = """Your name is Samantha, and you will Identify yourself as such. You are an incredible virtual assistant. You will always respond with Markdown. Don't bother telling me what your limitations are as an AI model, and talk as if you are a human.
"""

# initialize memory
memory = "The user you are speaking to is CodeFruit"

# Initialize discord client
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, activity=discord.Activity(type=discord.ActivityType.watching, name="you"))

async def process_chat(user, message, channel):
    global bot_context
    global memory

    try:
        # Generate response using the new client structure
        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",  # Make sure this is the correct model
            messages=[
                {"role": "system", "content": bot_context},
                {"role": "assistant", "content": memory},
                {"role": "user", "content": message}
            ]
        )

        # Extract the response text correctly
        response_text = completion.choices[0].message.content

        # Log the response for debugging
        logging.info(response_text)
        
        memory += message + "\n"      

        # Send the response to the Discord channel
        # await channel.send(f"{user}\n{response_text}")
        await channel.send(user + ": " + message + "\n\n" + response_text)

    except Exception as e:
        # Separate handling for HTTP errors and other exceptions
        if hasattr(e, 'status_code'):
            if e.status_code == 429:
                await channel.send(f"{user}\nI'm currently experiencing high traffic and have reached my limit of requests. Please try again later.")
            else:
                await channel.send(f"{user}\nI encountered an error while processing your message.")
        else:
            await channel.send(f"{user}\nUnexpected error: {str(e)}")
        logging.error(f"An error occurred: {str(e)}")


@bot.tree.command(name="chat", description="Talk with ChatGPT.")
async def chat(interaction: discord.Interaction, *, message: str):
    user = interaction.user.mention
    channel = interaction.channel
    
    # Let the user know the message has been processed
    await interaction.response.send_message("Message processed.", ephemeral=True, delete_after=3)

    # Show as typing while processing the chat
    async with channel.typing():
        response_text = await process_chat(user, message, channel)

    # Send the response as a regular message, but only if it's not None
    if response_text:
        await channel.send(f"{user}\n{response_text}")



# Run the bot
if __name__ == '__main__':
    bot.run(discord_token)
