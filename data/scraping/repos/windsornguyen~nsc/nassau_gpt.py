import os
import discord
from discord.ext import commands
from openai import OpenAI
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)
nassau_gpt = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

system_prompt = ()  # Enter your system prompt here


@bot.event
async def on_ready():
    logger.info('NassauGPT is online!')


@bot.event
async def on_message(message):
    if message.author.bot:
        return
    if message.channel.id != int(os.getenv('NASSAU_GPT_CHANNEL')):
        return
    if message.content.startswith('!'):
        return

    try:
        async with message.channel.typing():
            # Retrieve previous messages
            prev_messages = [msg async for msg in message.channel.history(limit=15)]
            prev_messages.reverse()

            conversation_log = [
                {
                    'role': 'system',
                    'content': system_prompt,
                }
            ]

            for msg in prev_messages:
                if msg.author.bot and msg.author.id != bot.user.id:
                    continue
                conversation_log.append(
                    {
                        'role': 'assistant' if msg.author.id == bot.user.id else 'user',
                        'content': msg.content,
                    }
                )

            response = nassau_gpt.chat.completions.create(
                model='gpt-4-1106-preview',
                messages=conversation_log,
                max_tokens=500,
                temperature=0.75,
                frequency_penalty=0.3,
                presence_penalty=0.3,
            )

            reply_content = response.choices[0].message.content.strip()

            # Truncation logic to coalesce response with Discord's message limit
            if len(reply_content) > 2000:
                # Find a natural ending point (like a full stop) before the limit
                end_index = reply_content.rfind('.', 0, 1997)
                if end_index == -1:
                    # If no natural ending, truncate normally
                    truncated_reply_content = reply_content[:1997] + '...'
                else:
                    truncated_reply_content = reply_content[: end_index + 1]
            else:
                truncated_reply_content = reply_content

        await message.reply(truncated_reply_content)

    except Exception as ex:
        logger.error(f'An error occurred: {ex}', exc_info=True)
        await message.channel.send(
            +'Sorry, I encountered a server-side issue while processing your message. Please give me a moment.'
        )


bot.run(os.getenv('NASSAU_GPT_TOKEN'))
