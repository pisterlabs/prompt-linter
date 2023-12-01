import tempfile
import time

import requests
import telepot
from openai import OpenAI
from telepot.loop import MessageLoop
from telepot.namedtuple import InlineKeyboardButton, InlineKeyboardMarkup

# Substitua pelo seu token da API do OpenAI
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# Inicialização do cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


# Substitua 'YOUR_TELEGRAM_TOKEN' pelo token do seu bot
TOKEN = "YOUR_TELEGRAM_TOKEN"
bot = telepot.Bot(TOKEN)

# Dictionary to maintain user state
user_state = {}


# Function to create size selection keyboard
def size_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="1024x1024", callback_data="1024x1024")],
            [InlineKeyboardButton(text="1024x1792", callback_data="1024x1792")],
            [InlineKeyboardButton(text="1792x1024", callback_data="1792x1024")],
        ]
    )
    return keyboard


# Function to create quality selection keyboard
def quality_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="HD", callback_data="hd")],
            [InlineKeyboardButton(text="Standard", callback_data="standard")],
        ]
    )
    return keyboard


# Function to handle chat messages
def on_chat_message(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type == "text":
        if msg["text"] == "/start":
            bot.sendMessage(
                chat_id,
                "*Dear user, please provide a description for the image you wish to generate. Include relevant details for the best result.*",
                parse_mode="markdown",
            )
        else:
            user_state[chat_id] = {"prompt": msg["text"]}
            bot.sendMessage(
                chat_id, "Choose the size of the image:", reply_markup=size_keyboard()
            )


def post_generation_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Generate another image with the same text",
                    callback_data="regenerate_same",
                )
            ],
            [
                InlineKeyboardButton(
                    text="Generate image with new text", callback_data="generate_new"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Generate with same text and new options",
                    callback_data="regenerate_options",
                )
            ],
        ]
    )
    return keyboard


# Function to handle callback queries
def on_callback_query(msg):
    query_id, from_id, query_data = telepot.glance(msg, flavor="callback_query")
    bot.answerCallbackQuery(query_id)

    chat_id = msg["message"]["chat"]["id"]
    message_id = msg["message"]["message_id"]

    if chat_id not in user_state:
        user_state[chat_id] = {}

    if "size" not in user_state[chat_id]:
        # Size selection
        user_state[chat_id]["size"] = query_data
        bot.sendMessage(
            chat_id, "Choose the quality of the image:", reply_markup=quality_keyboard()
        )
    elif (
        "quality" not in user_state[chat_id]
        and "awaiting_new_options" not in user_state[chat_id]
    ):
        # Quality selection
        user_state[chat_id]["quality"] = query_data

        # Generating the image
        bot.sendMessage(
            chat_id,
            "*Your request is being processed. Please wait...*",
            parse_mode="markdown",
        )
        generate_and_send_image(chat_id, user_state[chat_id])

        # Present options after sending the image
        bot.sendMessage(
            chat_id,
            "What would you like to do now?",
            reply_markup=post_generation_keyboard(),
        )

    elif query_data == "regenerate_same":
        # Regenerate image with the same text
        bot.sendMessage(
            chat_id,
            "*Generating another image with the same text. Please wait...*",
            parse_mode="markdown",
        )
        generate_and_send_image(chat_id, user_state[chat_id])
        bot.sendMessage(
            chat_id,
            "What would you like to do now?",
            reply_markup=post_generation_keyboard(),
        )

    elif query_data == "generate_new":
        # Request a new prompt
        bot.sendMessage(
            chat_id,
            "*Please provide a description for the image you wish to generate. Include relevant details for the best result.*",
            parse_mode="markdown",
        )
        user_state[chat_id] = {"prompt": None}  # Reset for a new prompt

    elif query_data == "regenerate_options":
        # The user wants to regenerate the image with the same text, but choose new options
        bot.sendMessage(
            chat_id,
            "Choose the new size for the image:",
            reply_markup=size_keyboard(),
        )
        user_state[chat_id] = {
            "prompt": user_state[chat_id]["prompt"],
            "awaiting_new_options": True,
        }

    elif "awaiting_new_options" in user_state[chat_id]:
        if "size" not in user_state[chat_id]:
            # New size selection
            user_state[chat_id]["size"] = query_data
            bot.sendMessage(
                chat_id,
                "Choose the new quality of the image:",
                reply_markup=quality_keyboard(),
            )
        else:
            # New quality selection
            user_state[chat_id]["quality"] = query_data
            bot.sendMessage(
                chat_id,
                "Generating image with the same text and new options. Please wait...",
                parse_mode="markdown",
            )
            generate_and_send_image(chat_id, user_state[chat_id])
            bot.sendMessage(
                chat_id,
                "What would you like to do now?",
                reply_markup=post_generation_keyboard(),
            )
            del user_state[chat_id]["awaiting_new_options"]


def generate_and_send_image(chat_id, user_data):
    # Generating the image with the DALL-E API
    response = client.images.generate(
        model="dall-e-3",
        prompt=user_data["prompt"],
        n=1,
        size=user_data["size"],
        quality=user_data["quality"],
    )

    # Get the URL of the generated image
    image_url = response.data[0].url

    # Download the image content
    image_content = requests.get(image_url).content

    # Saving the image content in a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        temp_image.write(image_content)
        temp_image.flush()

        # Send the photo to the user using the file path
        bot.sendPhoto(chat_id, photo=open(temp_image.name, "rb"))


# Message loop setup
MessageLoop(
    bot, {"chat": on_chat_message, "callback_query": on_callback_query}
).run_as_thread()

print("Running...")

# Keep the program running
while 1:
    time.sleep(10)
