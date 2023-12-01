from functools import wraps

from loguru import logger as log
from openai.error import RateLimitError
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from loader import config
from openai_api import openai_request


def send_typing_action(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    async def handler(update, context, *args, **kwargs):
        await context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
        return await func(update, context,  *args, **kwargs)

    return handler


@send_typing_action
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Send me any prompt to get OpenAI answer.")


@send_typing_action
async def message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id in map(int, config.TELEGRAM_USERS):
        try:
            log.trace(f"Received a prompt: {update.message.text}")
            response = openai_request(prompt=update.message.text)

        except RateLimitError as e:
            log.error(e)
            await update.message.reply_text(f"Something went wrong with OpenAI: `{e}`")

        else:
            log.trace(f"Received a response: {response}")

            await update.message.reply_text(
                f"`{update.message.text}` {response}",
                # parse_mode="MarkdownV2",
            )

    else:
        log.trace(f"User {update.effective_user.id} not in the list of allowed users")
        await update.message.reply_text("Access denied!")
