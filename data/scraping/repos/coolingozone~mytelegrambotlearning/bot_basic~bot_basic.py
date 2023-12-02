
#this is a basic telegram bot that serveas a template for future development
#This code simply echo back whatever user type and log the message
#last update: 12/7/2023
#Author: Teo Kok Keong

#aimport libraries
import os
import telebot
import openai
import logging
# Enable logging
logging.basicConfig(filename='example1.log',
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
bot = telebot.TeleBot("BOT TOKEN") #Replace by the coressponding bot token

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Hi, This is a test telegram bot that would echo your message! ")
    #logger.info("message is:  of %s", update.message.text)
@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    prompt = f"translate '{message.text}'  to chinese and present the results in line form stating first the  result,  and then the explanation in english, follow by an example sentence in chinese"
    logger.info("message is:  of  %s", message.text)
    bot.reply_to(message, message.text)

    
bot.infinity_polling()