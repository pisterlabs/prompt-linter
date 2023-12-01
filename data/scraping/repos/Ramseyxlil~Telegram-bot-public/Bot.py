#!/usr/bin/python

import openai
import telebot

# Set the API key
openai.api_key = "OPENAI API KEY"

# Create a Telegram bot
bot = telebot.TeleBot("TELEGRAM BOT TOKEN")

# Handle incoming messages
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(chat_id=message.chat.id, text="Hey i'm VUNA IKYK BOT. How can i help you today?")

@bot.message_handler(func=lambda message: True)
def generate_text(message):
  # Get user input
  text = message.text

  # Send request to OpenAI API
  response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=text,
    max_tokens=2000,
  )

  # Send response back to Telegram
  response_text = response.choices[0].text
  bot.send_message(chat_id=message.chat.id, text=response_text)

# Start the bot
bot.polling(none_stop=True)
