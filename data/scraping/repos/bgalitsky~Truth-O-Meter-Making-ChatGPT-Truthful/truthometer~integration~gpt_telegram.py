import telegram
from telegram.ext import CommandHandler, MessageHandler,  Updater #. Filters,
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"


# Define a function to handle incoming messages
def handle_message(update, context):
    # Get the user's message
    user_message = update.message.text

    # Send the user's message to ChatGPT
    response = openai.Completion.create(
        engine="davinci",
        prompt=user_message,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )

    # Get the response from ChatGPT
    chatbot_response = response.choices[0].text

    # Send the response back to the user
    update.message.reply_text(chatbot_response)


# Define a function to start the bot
def start_bot():
    # Set up the Telegram bot
    bot = telegram.Bot(token="YOUR_TELEGRAM_BOT_TOKEN")
    updater = Updater(bot.token, use_context=True)

    # Set up a message handler to handle incoming messages
    message_handler = MessageHandler(Filters.text & (~Filters.command), handle_message)
    updater.dispatcher.add_handler(message_handler)

    # Start the bot
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    start_bot()
