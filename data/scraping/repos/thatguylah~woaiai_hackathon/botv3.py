"""
Image-generating Bot that assists with image design ideas and image generation based on its own text-to-image prompt

Usage:
1. Sequence of automated questions to answer and then generating an image based on suggested image design.
2. Editing existing images to remove any object (inpainting) or extending out the image (outpainting)

Press Ctrl-C on the command line or send a signal to the process to stop the bot.
"""
import openai
import logging
from dotenv import dotenv_values
import requests
import json
import argparse
import os
from huggingface_hub import InferenceClient
from api.conversation import *
from api.inpainting import inpainting_handler
from api.outpainting import outpainting_handler

from telegram import __version__ as TG_VER
from telegram import (
    ForceReply,
    Update,
    ReplyKeyboardMarkup,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler,
    PicklePersistence,
)

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

# get config
config = dotenv_values(".env")

# get API tokens
HF_TOKEN = config["HF_API_KEY"]
openai.api_key = config["OPENAI_API_KEY"]

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(processName)s - %(threadName)s - [%(thread)d] - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
huggingFaceLogger = logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
messageHandlerLogger = logging.getLogger("telegram.bot").setLevel(logging.DEBUG)
applicationLogger = logging.getLogger("telegram.ext").setLevel(logging.DEBUG)

# assign variable name for each integer in sequence for easy tracking of conversation
(
    RESET_CHAT,
    VALIDATE_USER,
    USER_COMPANY,
    EDIT_COMPANY,
    IMAGE_TYPE,
    IMAGE_PURPOSE,
    SELECT_THEME,
    SELECT_IMAGE_DESIGN,
    CUSTOM_IMAGE_PROMPT,
    GENERATE_PROMPT_AND_IMAGE,
    GENERATE_IMAGE,
) = range(11)

# list of selected government agencies
lst_govt_agencies = [
    "Housing Development Board (HDB)",
    "Government Technology Agency (GovTech)",
    "Others",
]


# function to check bot's health status (CommandHandler type)
async def pong(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """serves a health check status for debugging

    Args:
        update (Update): _description_
        context (ContextTypes.DEFAULT_TYPE): _description_
    Returns:
        Pong to the user
    """
    await update.message.reply_text("Pong")


# function to start the bot
def main(dev_mode) -> None:
    if dev_mode:
        TELEBOT_TOKEN = config["TELEBOT_DEV_TOKEN"]
    else:
        TELEBOT_TOKEN = config["TELEBOT_TOKEN"]

    # create folders for outputs
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/image_output"):
        os.mkdir("data/image_output")

    # configure chatbot's persistence
    persistence = PicklePersistence(filepath="data/conversation")

    # create the Application pass telebot's token to application
    application = (
        Application.builder().token(TELEBOT_TOKEN).persistence(persistence).build()
    )

    # Conversation Handler with the states IMAGE_TYPE, IMAGE_PURPOSE, SELECTED_THEME, SELECTED_IMAGE_DESIGN
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start_command),
            CommandHandler("editcompany", edit_company_command),
            CommandHandler("choosetheme", get_previous_themes),
            CommandHandler("choosedesign", get_previous_image_designs),
            inpainting_handler,
            outpainting_handler,
        ],
        states={
            RESET_CHAT: [
                MessageHandler(
                    filters.Regex("(Generate Image Again)"), generate_image, block=False
                ),
                MessageHandler(
                    filters.Regex("(Generate New Image: Step-by-step Process)"),
                    validate_user,
                    block=False,
                ),
                MessageHandler(
                    filters.Regex("(Generate New Image: Use Custom Prompt)"),
                    get_user_custom_image_prompt,
                    block=False,
                ),
                MessageHandler(
                    filters.Regex("(Edit Existing Image)"), validate_user, block=False
                ),
            ],
            VALIDATE_USER: [
                MessageHandler(
                    filters.TEXT
                    & ~filters.Regex("(Generate Image: Use Custom Prompt)")
                    & ~filters.COMMAND,
                    validate_user,
                    block=False,
                ),
                MessageHandler(
                    filters.Regex("(Generate Image: Use Custom Prompt)"),
                    get_user_custom_image_prompt,
                    block=False,
                ),
            ],
            USER_COMPANY: [
                MessageHandler(
                    filters.TEXT
                    & ~filters.Regex(
                        "(Edit Existing Image|Generate Image: Use Custom Prompt|Yes|No)"
                    )
                    & ~filters.COMMAND,
                    get_user_company,
                    block=False,
                ),
                MessageHandler(
                    filters.Regex("(Edit Existing Image)"), validate_user, block=False
                ),
                MessageHandler(filters.Regex("(Yes)"), get_user_company, block=False),
                MessageHandler(filters.Regex("(No)"), validate_user, block=False),
                MessageHandler(
                    filters.Regex("(Generate Image: Use Custom Prompt)"),
                    get_user_custom_image_prompt,
                    block=False,
                ),
            ],
            IMAGE_TYPE: [
                MessageHandler(
                    filters.Regex("(Continue)")
                    & ~filters.Regex("(/quit|Edit Company Name)"),
                    get_image_type,
                    block=False,
                ),
                MessageHandler(
                    filters.Regex("(Edit Company Name)"),
                    edit_company_command,
                    block=False,
                ),
            ],
            IMAGE_PURPOSE: [
                MessageHandler(
                    filters.Regex("(Poster|Realistic Photo|Illustration)"),
                    get_image_purpose,
                    block=False,
                )
            ],
            SELECT_THEME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, get_theme, block=False)
            ],
            SELECT_IMAGE_DESIGN: [
                MessageHandler(
                    filters.TEXT
                    & ~filters.Regex("(Propose other themes|Write own theme)")
                    & ~filters.COMMAND,
                    select_image_design,
                    block=True,
                ),
                MessageHandler(
                    filters.Regex("(Propose other themes)"), get_theme, block=False
                ),
                MessageHandler(
                    filters.Regex("(Write own theme)"),
                    get_user_custom_theme,
                    block=False,
                ),
            ],
            CUSTOM_IMAGE_PROMPT: [
                MessageHandler(
                    filters.Regex("(Generate Image: Use Custom Prompt|Continue)"),
                    get_user_custom_image_prompt,
                    block=False,
                )
            ],
            GENERATE_PROMPT_AND_IMAGE: [
                MessageHandler(
                    filters.TEXT
                    & ~filters.Regex(
                        "(Propose other image designs|Write own image design)"
                    )
                    & ~filters.COMMAND,
                    generate_prompt_and_image,
                    block=False,
                ),
                MessageHandler(
                    filters.Regex("(Propose other image designs)"),
                    select_image_design,
                    block=False,
                ),
                MessageHandler(
                    filters.Regex("(Write own image design)"),
                    get_user_custom_image_design,
                    block=False,
                ),
            ],
            GENERATE_IMAGE: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, generate_image, block=False
                )
            ],
        },
        fallbacks=[CommandHandler("quit", quit_command)],
        allow_reentry=True,  # allow user to enter back any state of the ConversationHandler
        name="ImageGeneratingBot",
        persistent=True,
        block=False,
    )

    # handler to check bot's health status
    ping_handler = CommandHandler("ping", pong, block=False)

    # add handlers to application
    application.add_handler(conv_handler)
    application.add_handler(ping_handler)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-DEV", "--dev", action="store_true", help="Run with local Tele API token"
    )
    args = parser.parse_args()
    main(args.dev)
