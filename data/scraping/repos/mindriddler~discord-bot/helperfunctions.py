# import shutil
import datetime
import json
import logging
import logging.handlers
from io import BytesIO
import os
import jsonschema

import requests
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg
from config_validation_schema import schema as CONFIG_SCHEMA
from jsonschema import validate
from loguru import logger as tmp_logger

# CONSTANTS

HELP_STR = """
Help section for ChatGPT Discord Bot\n
/chatgpt - Start a conversation with ChatGPT
/dm - Start a private conversation with ChatGPT
/about - Brief description
/help - This help message
"""

ABOUT_STR = """
For now the bot till NOT remember past conversations.
It is something i am working on getting done.
"""

LOG_FOLDER = "logs"

DEFAULT_THREAD_MESSAGE = """
Hey! I have created this thread for us to have a public conversation.
If you want to have a private conversation with me use the command '!dm',
otherwise state your question and i will gladly help you!
"""

DEFAULT_DM_MESSAGE = (
    """this is a DM from ChatGPT. You can now have a conversation with me here and it will be our little secret."""
)

DISCLAIMER = """
DISCLAIMER:
I do not have a memory yet. I wont be able to remember our conversation.
Threads will be automatically closed after 24 hours.
"""

COMMAND_DESCRIPTIONS = {
    "schedule":
        "Shows a schedule for n about of days",
    "chatgpt":
        "Ask ChatGPT a question of your choice and you will get a answer back.",
    "help":
        "Show a small help section with available commands.",
    "about":
        "Show a small about section for the bot.",
    "dm":
        "ChatGPT will start DMs with you.",
    "image":
        "Generate a image using the DALL-E AI model from OpenAI.",
    "github stats":
        "Will show you brief github stats for specified user",
}


async def convert_svg_url_to_png(*svg_urls, suppress_warnings):
    pngs = []
    if suppress_warnings:
        logging.getLogger("svglib.svglib").setLevel(logging.CRITICAL)
        logging.getLogger("reportlab.graphics").setLevel(logging.CRITICAL)

    for svg_url in svg_urls:
        response = requests.get(svg_url)
        svg_data = BytesIO(response.content)
        drawing = svg2rlg(svg_data)
        img_data = BytesIO()
        renderPM.drawToFile(drawing, img_data, fmt="PNG")
        img_data.seek(0)
        pngs.append(img_data)

    return pngs


def split_message(message, max_length=2000):
    if len(message) <= max_length:
        return [message]
    messages = []
    current_message = ""
    for word in message.split():
        if len(current_message) + len(word) + 1 <= max_length:
            current_message += f" {word}"
        else:
            messages.append(current_message)
            current_message = f"{word}"
    messages.append(current_message)
    return messages


def openailoghandler():
    filename = read_config("logger")
    logger = logging.getLogger("openai")
    logger.setLevel(getattr(logging, filename["log_level_openai"]))

    # Format the filename using the current time
    # This has to be done in order to get the same type of filename for all the log files
    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    filename = filename["log_path_openai"].replace(
        "{time:YYYY-MM-DD-HH-mm-ss!UTC}", current_time)

    handler = logging.handlers.RotatingFileHandler(
        filename=filename,
        encoding="utf-8",
        maxBytes=32 * 1024 * 1024,
        backupCount=5,
    )
    dt_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        "[{asctime}] [{levelname:<8}] {name}: {message}", dt_fmt, style="{")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def discordloghandler():
    filename = read_config("discord")
    logger = logging.getLogger("discord")
    logger.setLevel(getattr(logging, filename["log_level_discord"]))

    # Format the filename using the current time
    # This has to be done in order to get the same type of filename for all the log files
    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    filename = filename["log_path_discord"].replace(
        "{time:YYYY-MM-DD-HH-mm-ss!UTC}", current_time)

    handler = logging.handlers.RotatingFileHandler(
        filename=filename,
        encoding="utf-8",
        maxBytes=32 * 1024 * 1024,
        backupCount=5,
    )
    dt_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        "[{asctime}] [{levelname:<8}] {name}: {message}", dt_fmt, style="{")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def read_config(section):
    """
    Returns specified section of config.json as a dictionary
    """
    tmp_logger.info(f"Reading config for '{section}'")
    default_path = f"{get_bot_directory()}/config/config.json"

    tmp_logger.debug(default_path)
    try:
        with open(os.environ.get("BOT_CONFIG_FILE", default_path),
                  "r",
                  encoding="utf-8") as file:
            config_data = json.load(file)
            tmp_logger.debug(config_data)
            validate_config(config_data, schema=CONFIG_SCHEMA)
            if section in config_data:
                tmp_logger.debug(config_data[section])
                return config_data[section]
            else:
                raise ValueError(
                    f"Section '{section}' not found in config file")
    except (json.decoder.JSONDecodeError,
            jsonschema.exceptions.ValidationError) as error:
        tmp_logger.error(error)
        raise


def get_bot_directory() -> str:
    """
    Returns the directory that bot.py is in.
    """

    return os.path.dirname(os.path.realpath(__file__))


def validate_config(config: dict, schema: dict) -> str | None:
    """
    Check that all required fields are present in the configuration file

    Args:
        config (dict): The configuration file
        schema (dict): The schema to validate against

    Returns:
        None

    Raises:
        jsonschema.exceptions.ValidationError: If the configuration file is missing required fields
    """

    validate(instance=config, schema=schema)
