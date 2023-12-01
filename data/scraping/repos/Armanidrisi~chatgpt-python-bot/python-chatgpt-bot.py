#Author: Arman Idrisi

import telebot
import openai


#Bot Api Token
API_TOKEN = ''
#Openai Api Key
openai.api_key=""


bot = telebot.TeleBot(API_TOKEN)

#Generate The Response
def get_response(msg):
	completion = openai.Completion.create(
    engine="text-davinci-003",
    prompt=msg,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)
	return completion.choices[0].text

# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
	 # bot.send_message(message.chat.id,message.text)
	   bot.send_message(message.chat.id, """\
Hi there, I am A Ai ChatBot.

I am here to Give Answers Of Your Question.

I Am Created Using Chatgpt Api ! 

Use /ask  To Ask Questions\
""")

#Handle The '/ask'
@bot.message_handler(commands=['ask'])
def first_process(message):
	bot.send_message(message.chat.id,"Send Me your Question")
	bot.register_next_step_handler(message,second_process)
def again_send(message):
  bot.register_next_step_handler(message,second_process)
def second_process(message):
  bot.send_message(message.chat.id,get_response(message.text))
  again_send(message)

 
bot.infinity_polling()

