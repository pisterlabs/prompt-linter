# System
from pathlib import Path

# Telegram libraries
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Info files
from bot_info import TOKEN, BOT_NAME

# Text processing
import text_processing as tp

# OpenAI and generation
import openai
from openai import OpenAI
client = None
speech_file_path = Path(__file__).parent / 'audio.mp3'
api_key_path = 'api_key.txt'

# Commands to interact with the bot
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Hello, I'm {BOT_NAME}! I am a TTS agent for Sergiy Horef. Type /help to see what I can do.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please enter your text so I can provide you with a recording on it.")

# Audio generation
def generate_audio(text: str) -> None:
    response = client.audio.speech.create(
    model="tts-1-hd",
    voice="onyx",
    input=text
    )
    response.stream_to_file(speech_file_path)

# Logic to handle responses
def handle_response(text: str) -> str:
    response: str = tp.preprocess(text)
    return response

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    print(message)
    if update.message == None:
        message = update.channel_post
        print(message)
    text: str = message.text

    response: str = handle_response(text)
    generate_audio(response)

    await message.reply_audio(audio=open('audio.mp3', 'rb'))

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')
    await update.message.reply_text(f'Error: {context.error}')

if __name__ == '__main__':
    # Setting up the API key.
    openai.api_key_path = api_key_path
    client = OpenAI(api_key=open(api_key_path, 'r').read())

    # Setting up the bot.
    print('Starting bot...')
    app = Application.builder().token(TOKEN).build()

    # Commands handler for the bot
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    # Message handler for the bot
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Error handler for the bot
    app.add_error_handler(error)

    # Polling the bot
    print('Polling...')
    app.run_polling(poll_interval=3)