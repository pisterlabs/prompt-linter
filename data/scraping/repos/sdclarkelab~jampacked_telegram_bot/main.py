import os
from dotenv import load_dotenv

from typing import Final
import openai
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

BOT_USERNAME: Final = '@jampacked_bot'
TELEGRAM_TOKEN: Final = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY: Final = os.getenv("OPENAI_API_KEY")

# Set openai key
openai.api_key = OPENAI_API_KEY


# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Wah Gwan! (Hello!). Which data and Jamaican city would you like to travel to?\n'
                                    'Example: Will Ochi be packed this weekend?')


# Responses
def handle_response(user_message: str) -> str:
    data_source = """
    {
        "events": [
            {
                "location": {
                    "city": "Ocho Rios",
                    "country": "Jamaica"
                },
                "event_name": "Reggae Sumfest",
                "start_date": "2023-07-18",
                "end_date": "2023-07-18",
                "type": "party"
            },
            {
                "location": {
                    "city": "Montego Bay",
                    "country": "Jamaica"
                },
                "event_name": "Some Cool Party",
                "start_date": "2023-07-20",
                "end_date": "2023-07-20",
                "type": "party"
            }
            {
                "location": {
                    "city": "Ocho Rios",
                    "country": "Jamaica"
                },
                "ship_name": "Carnival Horizon",
                "arrival_date": "2023-07-18T08:00:00Z",
                "departure_date": "2023-07-18T17:00:00Z",
                "type": "cruise"
            }
        ]
    }
    """

    chat_prompt: str = f'{data_source} \n {user_message}'
    response_from_chatgpt = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                         messages=[{'role': "user", 'content': chat_prompt}])
    return response_from_chatgpt.choices[0].message.content


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message: str = update.message.text
    message_type: str = update.message.chat.type

    print(f'User ({update.message.chat.id}) in {message_type}: "{message}"')

    if message_type == 'group' or message_type == 'supergroup':
        if BOT_USERNAME in message:
            message: str = message.replace(BOT_USERNAME, '').strip()
        else:
            return

    response: str = handle_response(message)
    print('Bot response: ', response)
    await update.message.reply_text(response)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


if __name__ == '__main__':
    print('Starting bot ...')
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))

    app.add_error_handler(error)

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print('Polling ...')
    app.run_polling(poll_interval=3)
