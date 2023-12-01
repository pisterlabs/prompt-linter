import os
import openai
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
load_dotenv('data.env')

# Получение токенов из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Инициализация клиентов Telegram и OpenAI
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
openai.api_key = OPENAI_API_KEY

# Обработчик команды /start
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Привет! Я ChatGPT, чем я могу вам помочь?")

# Обработчик текстовых сообщений
def generate_message(update, context):
    prompt = update.message.text
    message = generate_text(prompt)
    context.bot.send_message(chat_id=update.effective_chat.id, text=message)

# Функция для генерации текста с помощью OpenAI
def generate_text(prompt):
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt,
      max_tokens=60,
      n=1,
      stop=None,
      temperature=0.5,
    )
    message = response.choices[0].text.strip()
    return message

# Создание объекта для обработки событий от Telegram
updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)

# Получение диспетчера для регистрации обработчиков
dispatcher = updater.dispatcher

# Регистрация обработчиков
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, generate_message))

# Запуск бота
updater.start_polling()
