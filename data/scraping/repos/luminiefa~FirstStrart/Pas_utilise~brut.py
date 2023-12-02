import logging
import mariadb
import pytesseract
from telegram import __version__ as TG_VER
from telegram import ForceReply, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from PIL import Image
import os
import requests

from dotenv import dotenv_values
import openai

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Model
class Database:
    def __init__(self):
        self.connection = mariadb.connect(
            user="root",
            password="",
            host="localhost",
            port=3306,
            database="FirstStrat",
        )
        self.cursor = self.connection.cursor()

    def insert_text(self, text):
        sql = "INSERT INTO texts (text) VALUES (?)"
        self.cursor.execute(sql, (text,))
        self.connection.commit()


# View
class BotView:
    def __init__(self, application, chatcompletion_api_key):
        self.application = application
        self.chatcompletion_api_key = chatcompletion_api_key

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        user = update.effective_user
        await update.message.reply_html(
            rf"Hi {user.mention_html()}!",
            reply_markup=ForceReply(selective=True),
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        await update.message.reply_text("Help!")

    async def echo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Echo the user message."""
        message = update.message.text
        await self.process_message(message, update)

    async def process_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process the received image."""
        photo = update.message.photo[-1]  # Get the last (highest resolution) photo
        file = await photo.get_file()

        # Specify the path where you want to save the image
        current_directory = os.path.abspath(os.getcwd())
        image_directory = os.path.join(current_directory, "images")
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)

        filename = "image.jpg"
        file_path = os.path.join(image_directory, filename)

        try:
            await file.download_to_drive(file_path)
            text = self.extract_text_from_image(file_path)
            database = Database()
            database.insert_text(text)

            # Send message to ChatCompletion API
            api_key = self.chatcompletion_api_key
            prompt = "Your prompt here: " + "fais un tri qui te semble le plus logique et ignore les infos manquantes" + text  # Use the extracted text as the prompt

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                api_key=api_key,
            )

            answer = response.choices[0].message.content

            # Insert ChatCompletion response to the database
            database.insert_text(answer)

            await update.message.reply_text(answer)
            await update.message.reply_text("Text extracted and saved to the database.")
        except Exception as e:
            await update.message.reply_text(f"Error processing the image: {str(e)}")

    def extract_text_from_image(self, image_path):
        """Extract text from the given image using OCR."""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text


# Controller
class BotController:
    def __init__(self, application, chatcompletion_api_key):
        self.view = BotView(application, chatcompletion_api_key)

    def register_handlers(self, application):
        application.add_handler(CommandHandler("start", self.view.start))
        application.add_handler(CommandHandler("help", self.view.help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.view.echo))
        application.add_handler(MessageHandler(filters.PHOTO, self.view.process_image))


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    env_values = dotenv_values(".env")
    token = env_values["TOKEN"]

    application = Application.builder().token(token).build()

    # Get your ChatCompletion API key from environment variables or .env file
    chatcompletion_api_key = env_values["chatgpt_key"]

    controller = BotController(application, chatcompletion_api_key)
    controller.register_handlers(application)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
