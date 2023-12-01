import os
import telebot
import dotenv
import openai
from loguru import logger


dotenv.load_dotenv()

bot = telebot.TeleBot(os.getenv("BOT_TOKEN"))
openai.api_key = os.getenv("OPENAI_API_KEY")


@bot.message_handler(commands=['help'])
def send_help(message):
    logger.info(message.__dict__["json"]
    bot.reply_to(message,
                 "Para usar o bot, basta mandar uma mensagem com o texto que você quer escrever e eu vou te ajudar a completá-lo. Você também pode usar o comando /ask para fazer perguntas ao bot.")


@bot.message_handler(commands=['start'])
def send_welcome(message):
    logger.info(message.__dict__["json"])
    bot.reply_to(message,
                 "Olá, como vai? Eu sou o Socos Bot, e estou aqui para te ajudar a escrever textos. Basta me mandar uma mensagem com o texto que você quer escrever e eu vou te ajudar a completá-lo.")


@bot.message_handler(commands=['ask'])
def ask_bot(message):
    logger.info(message.__dict__["json"])

    bot.reply_to(message, "Aguarde um momento...")

    prompt = message.text.replace("/ask", "")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=1024,
        n=1,
        stop=None,
    )
    bot.reply_to(message, response.choices[0].text)


logger.info("Starting bot...")
bot.infinity_polling()
