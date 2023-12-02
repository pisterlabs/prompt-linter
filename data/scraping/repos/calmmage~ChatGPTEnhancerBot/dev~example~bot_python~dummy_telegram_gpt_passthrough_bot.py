"""a simple bot that just forwards queries to openai and sends the response"""

import logging
from functools import wraps

import openai
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Load the secrets from a file
secrets = {}
with open("secrets.txt", "r") as f:
    for line in f:
        key, value = line.strip().split(":", 1)
        secrets[key] = value

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def telegram_decorator(func, **kwargs):
    @wraps(func)
    def wrapper(update: Update, context: CallbackContext) -> None:
        response = func(update.message.text, **kwargs)
        update.message.reply_text(response)

    return wrapper


# openai.organization = "org-cSwRU2HIBymBxEijKOapuNID"
openai.api_key = secrets["openai_api_key"]


# openai.Model.list()

def chatbot(prompt, model='text-ada:001', max_tokens=500,
            **kwargs):
    """
    https://beta.openai.com/docs/api-reference/completions/create

    :param prompt:
    :param model: For testing purposes - cheap - 'text-ada:001'. For real purposes - "text-davinci-003" - expensive!
    :param temperature:
    :param max_tokens:
    :param top_p:
    :param n:
    :param stream:
    :param stop:
    :param kwargs:
    :return:
    """
    # Send the message to the OpenAI API
    response = openai.Completion.create(model=model, prompt=prompt, max_tokens=max_tokens,
                                        **kwargs)

    # Extract the response from the API response
    response_text = response['choices'][0]['text']

    # Return the response to the user
    return response_text


def main(expensive: bool) -> None:
    """
    Start the bot
    :param expensive: Use 'text-davinci-003' model instead of 'text-ada:001'
    :return:
    """
    # Create the Updater and pass it your bot's token.
    token = secrets["telegram_api_token"]
    updater = Updater(token)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    model = "text-davinci-003" if expensive else "text-ada:001"
    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command,
                                          telegram_decorator(chatbot, model=model)))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expensive", action="store_true",
                        help="use expensive calculation - 'text-davinci-003' model instead of 'text-ada:001' ")
    args = parser.parse_args()

    main(expensive=args.expensive)
