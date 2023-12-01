from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import telegram
import openai
import config

from messages import (
    OPENAI_TIMEOUT_ERROR_MESSAGE,
    UNKNOWN_ERROR_MESSAGE
)

logging = config.logging


def handle_error(update, context):
    logging.debug("Entering handle_error")

    # Creating Inline Keyboard Markup for the 'retry' button at the bottom of the error message
    inline_keyboard = [[InlineKeyboardButton('Retry', callback_data='retry')]]
    reply_markup = InlineKeyboardMarkup(inline_keyboard)

    # Extract the occured error from context
    error = context.error

    # Handle OpenAI Timeout Error
    if isinstance(error, openai.error.Timeout):
        error_message = context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=OPENAI_TIMEOUT_ERROR_MESSAGE.format(error),
            reply_markup=reply_markup
        )
        context.user_data["last_error_message"] = error_message
        logging.error(f"OpenAI Timeout Error: {error}", exc_info=True)
    # Handle Network Error (internet issues)
    elif isinstance(error, telegram.error.NetworkError):
        logging.error(f"Telegram NetworkError: {error}", exc_info=True)
    # Handle all other errors
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=UNKNOWN_ERROR_MESSAGE.format(error)
        )
        logging.error(f"Unrecognized Error: {error}", exc_info=True)

    logging.debug("Exiting handle_error")
