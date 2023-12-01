from enum import Enum
import logging
import time
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.ext import (
    filters, 
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    CallbackQueryHandler,
    Application,
    ContextTypes
)
import openai
from logger import UserActionLogger

WELCOME = "WELCOME"
IMAGE_GEN = "IMAGE_GEN"
IMAGE_SIZE = "IMAGE_SIZE"
IMAGE_PROMPT = "IMAGE_PROMPT"
IMAGE_N = "N"
SELECTING_LEVEL = "SELECTING_LEVEL"
TYPING_PROMPT = "TYPING_PROMPT"
TYPING_N = "TYPING_N"
ASK_FOR_N = "ASK_FOR_N"
SELECTING_SIZE = "SELECTING_SIZE"
ASK_FOR_PROMPT = "ASK_FOR_PROMPT"
BACK_TO_START = "BACK_TO_START"
END_IMAGEGEN = "END_IMAGEGEN"

BACK_TO_MENU_COMMAND = "back_to_main"
IMAGE_GEN_START_COMMAND = "imagegen"
RESTART_IMAGE_GEN_COMMAND = "restart_imagegen"

class ImageSize(Enum):
    SMALL = 256
    MEDIUM = 512
    LARGE = 1024

class ImageGen:
    def __init__(self, update:Update, context: ContextTypes.DEFAULT_TYPE, username:str, id:str, application:Application, logger:UserActionLogger):
        self.update = update
        self.context = context
        self.username = username
        self.id = id
        self.application = application
        self.logger = logger
        self.imagegen_handlers = []
        self.welcome = False
        self.last_back_message_id = None
    
    async def run(self):
        self.logger.log_inline('%s', self.username, self.id, IMAGE_GEN)

        image_gen_welcome_text = (
            "You are now using the Image Generation tool!\n"
            f"Type /{IMAGE_GEN_START_COMMAND} to get started."
        )
        if (self.welcome == False):
            await self.update.callback_query.edit_message_text(text=image_gen_welcome_text)
            self.welcome = True
        await self.add_image_handlers()
            
    async def add_image_handlers(self):
        image_gen_entry_handler = CommandHandler(f"{IMAGE_GEN_START_COMMAND}", self.image_gen_entry)
        ask_for_n_handler = CallbackQueryHandler(self.ask_for_n, pattern=f"^{ImageSize.SMALL.value}$|^{ImageSize.MEDIUM.value}$|^{ImageSize.LARGE.value}$")
        save_n_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, self.save_n)
        save_image_prompt_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, self.save_image_prompt)
        restart = CommandHandler(f"{RESTART_IMAGE_GEN_COMMAND}", self.restart)
        
        # Image Conversation
        image_gen_conv = ConversationHandler(
            entry_points=[image_gen_entry_handler],
            states= {
                SELECTING_SIZE:[
                  ask_for_n_handler
                ],
                TYPING_N: [
                  save_n_handler
                ],
                TYPING_PROMPT: [
                  save_image_prompt_handler
                ]
            },
            fallbacks = [
                restart
            ]
        )
        self.imagegen_handlers.append(image_gen_conv)
        self.application.add_handler(image_gen_conv)

    async def image_gen_entry(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
        self.logger.log_action('%s', self.username, self.id, '/imagegen')

        """Selecting Image Size"""
        image_size_text = (
            "Select an image size for the generated photo"
        )
        image_size_buttons = [
            [        
                InlineKeyboardButton(text="256", callback_data=str(ImageSize.SMALL.value)),
                InlineKeyboardButton(text="512", callback_data=str(ImageSize.MEDIUM.value)),
                InlineKeyboardButton(text="1024", callback_data=str(ImageSize.LARGE.value))
            ],
        ]
        image_size_keyboard = InlineKeyboardMarkup(image_size_buttons)
        await context.bot.send_message(chat_id=self.id, text=image_size_text, reply_markup=image_size_keyboard)
        return SELECTING_SIZE

    # Saves the image prompt typed in by user, transits to image generation state
    async def ask_for_n(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
        context.user_data[IMAGE_SIZE] = update.callback_query.data
        self.logger.log_inline('%s', self.username, self.id, self.get_image_size())
        
        image_prompt_text = (
            f"Size of image: {self.get_image_size()}\n"
            f"Type in the number (1-10) of images to generate.\n"
        )
        await update.callback_query.edit_message_text(text=image_prompt_text)
        return TYPING_N
    
    async def save_n(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.logger.log_action('%s', self.username, self.id, update.message.text)
        try:
            n = update.message.text
            if int(n) > 10 or int(n) < 1:
                await context.bot.send_message(chat_id=self.id, text="Please key in a valid number (1-10)")
                return TYPING_N
            context.user_data[IMAGE_N] = n
            await self.ask_for_image_prompt(update=self.update, context=self.context)
            return TYPING_PROMPT
        except ValueError:
            await context.bot.send_message(chat_id=self.id, text="Please key in a number (1-10)")
            return TYPING_N

    # Saves the image prompt typed in by user, transits to image generation state
    async def ask_for_image_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
        image_prompt_text = (
            f"Size of image: {self.get_image_size()}\n"
            f"Number of images to generate: {self.get_n()}\n" 
            f"Type in your prompt for the image \n"
        )
        await context.bot.send_message(chat_id=self.id, text=image_prompt_text)
        return TYPING_PROMPT

    async def save_image_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.logger.log_action('%s', self.username, self.id, update.message.text)
        try:
            context.user_data[IMAGE_PROMPT] = update.message.text
            return await self.gen_image()
        
        # Prevent the user from typing into the text box when choosing options
        except KeyError as e:
            await self.context.bot.send_message(chat_id=self.id, text="Please only select options from the messages")
            return await self.image_gen_entry(update=self.update, context=self.context)
    
    def get_image_size(self):
        image_size_string = self.context.user_data[IMAGE_SIZE]
        image_size_param = f"{image_size_string}x{image_size_string}"
        return image_size_param
    
    def get_image_prompt(self):
        image_prompt = self.context.user_data[IMAGE_PROMPT]
        return image_prompt

    def get_n(self):
        image_n = self.context.user_data[IMAGE_N]
        return int(image_n)

    async def gen_image(self):
        self.logger.log_bot('%s', self.username, self.id, f'generating ChatGPT image - {self.get_image_prompt()}')
        try:
            image_finish_text = (
                "If you want to generate more images with a different prompt using the same image size and number, just type the prompt below.\n" 
                f"Type /{RESTART_IMAGE_GEN_COMMAND} to redo the whole settings selection.\n"
                f"Type /{BACK_TO_MENU_COMMAND} to return to the main menu."
            )
            chosen_image_settings_text = (
                "Generating photos with the following settings \n"
                f"Size of image: {self.get_image_size()}\n"
                f"Number of images: {self.get_n()}\n" 
                f"Prompt: {self.get_image_prompt()}"            
            )
            if (self.last_back_message_id != None):
                await self.context.bot.delete_message(chat_id=self.update.effective_chat.id, message_id=self.last_back_message_id)
            await self.context.bot.send_message(chat_id=self.id, text=chosen_image_settings_text)
            result = openai.Image.create(
                prompt=self.get_image_prompt(),
                n=self.get_n(),
                size= self.get_image_size()
                )
            for photo in result.data:
                await self.context.bot.send_photo(chat_id=self.id, photo=photo.url)
            self.logger.log_bot('%s', self.username, self.id, 'finished image generation' )
            await self.context.bot.send_message(chat_id=self.id, text=image_finish_text)
        except openai.InvalidRequestError as e:
            print(e)
            
    
    async def restart(self, update:Update, context: ContextTypes.DEFAULT_TYPE):
        self.logger.log_action('%s', self.username, self.id, '/restart_imagegen')
        return await self.image_gen_entry(update=self.update, context=self.context)

    async def remove_image_handlers(self):
        for handler in self.imagegen_handlers:
            self.application.remove_handler(handler=handler)
        return 

    

    
