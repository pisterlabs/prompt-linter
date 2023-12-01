#!/usr/bin/env python
# pylint: disable=unused-argument, wrong-import-position
# This program is dedicated to the public domain under the CC0 license.

from llama_index import VectorStoreIndex, ListIndex, StorageContext, load_index_from_storage
from llama_index.indices.composability import ComposableGraph
from langchain.chat_models import ChatOpenAI
import logging
import os
import openai
import sys

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

LLAMA_INDEX_ROOT_DIR = SCRIPT_DIR + "/llama_index"

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai.api_key = OPENAI_API_KEY;


llama_index_subdirs = []
for entry in os.walk(LLAMA_INDEX_ROOT_DIR):
    if "index_store.json" in entry[2]:
        llama_index_subdirs.append(entry[0])

print("Using Llama Indices from:")
for subdir in llama_index_subdirs:
    print("  - " + subdir)

def build_llama_query_engine():
    if len(llama_index_subdirs) == 1:
        ll_storage_context = StorageContext.from_defaults(persist_dir=llama_index_subdirs[0])
        ll_index = load_index_from_storage(ll_storage_context)
        return ll_index.as_query_engine()
    else:
        ll_indices = []
        for subdir in llama_index_subdirs:
            ll_storage_context = StorageContext.from_defaults(persist_dir=subdir)
            ll_index = load_index_from_storage(ll_storage_context)
            ll_indices.append(ll_index)

        ll_graph = ComposableGraph.from_indices(ListIndex, ll_indices, index_summaries=[""] * len(ll_indices))
        return ll_graph.as_query_engine()

ll_query_engine = build_llama_query_engine()


async def get_chatgpt_response(prompt):
    response = ll_query_engine.query(prompt)

    return response.response


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    response_text = await get_chatgpt_response(update.message.text)

    await update.message.reply_text(response_text)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
