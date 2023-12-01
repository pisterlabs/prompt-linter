from bot.bot_instance.bot import bot_instance
import telebot
from services.openai_service import OpenAiService
from bot.handlers.shared import tg_exception_handler

openai_service = OpenAiService()

@bot_instance.message_handler(commands=['clear_gpt_context'])
@tg_exception_handler
def clr_handler(message: telebot.types.Message):
    openai_service.clear_context(str(message.from_user.id))
    bot_instance.reply_to(message, "Context cleared.")