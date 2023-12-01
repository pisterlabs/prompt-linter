import openai
from telebot import TeleBot
from telebot import types
import os
from alive1 import keep_alive
 
openai.api_key = os.getenv("ai_key")

TOKEN = os.getenv("tg_key")
bot = TeleBot(TOKEN)

keep_alive()
 
#If command not supported bot will send message about it
def handle_command(chat_id, text):
    bot.send_message(chat_id, "Command not supported.")
 

#Function that reacts on command start
@bot.message_handler(commands=['start'])
def start(message):
    welcome_message = "Hello," + message.from_user.first_name + ' ' + message.from_user.last_name
    info = 'This is ChatGPT in Telegram, what is your question?'
    bot.send_message(message.chat.id, welcome_message)
    bot.send_message(message.chat.id, info)


#Fuction that reacts on any text user typed
@bot.message_handler(content_types=['text'])
def handle_text(message):
    
    chat_id = message.chat.id
    text = message.text.strip().lower()

    # Check if the message is a command
    if text.startswith('/'):
        # Process commands
        handle_command(chat_id, text)
    else:
        # Process user messages
        handle_user_message(chat_id, text)

#Functions that handles information about message
def handle_user_message(chat_id, text):
    # Call the OpenAI ChatGPT API to get a response
    response = chat_with_gpt(text)

    # Send the response to the user
    bot.send_message(chat_id, response)


# Call the OpenAI ChatGPT API
def chat_with_gpt(text):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=text,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5
    )

    return response.choices[0].text.strip()

#Bot looks for updates
bot.polling(none_stop=True)
