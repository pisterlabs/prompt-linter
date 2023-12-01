import os
import logging

import azure.functions as func
import telebot

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from utils import detect_lang

api_key = os.getenv("DUTCH_OPENAPI_KEY")
bot_key = os.getenv("BOT_KEY")

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
bot = telebot.TeleBot(bot_key)
llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo", temperature=0)

def translate_msg(msg: str) -> str:
    """Translates the given message from English to Russian or vice versa."""
    curr_lang = detect_lang(msg)

    if curr_lang == "nl":
        translation_lang = "english"
        current_lang = "dutch"
    else:
        translation_lang = "dutch"
        current_lang = "english"

    prompt_template = PromptTemplate.from_template("Translate the message from {current_lang} to {translation_lang}."
                                                   "Message: {msg}")
    prompt = prompt_template.format(current_lang=current_lang, translation_lang=translation_lang, msg=msg)

    return llm.predict(prompt)


@bot.message_handler(func=lambda message: True, content_types=['text'])
def echo_all(message: telebot.types.Message) -> None:
    """Handles all messages that are not commands."""
    response = translate_msg(message.text)
    bot.reply_to(message, response)


@bot.message_handler(commands=['start'])
def start_message(message: telebot.types.Message) -> None:
    bot.send_message(message.chat.id, """Hello! I am your OpenAI-based translator bot.
                      I translate English to Russian and vice versa!""")


@bot.message_handler(commands=['help'])
def help_message(message: telebot.types.Message) -> None:
    bot.send_message(message.chat.id, """I am your OpenAI-based translator bot.
                      Just tag me in your message and I will translate it for you!""")


@app.route(route="tlg")
def my_dutch_bot(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    request_body_dict = req.get_json()
    update = telebot.types.Update.de_json(request_body_dict)
    bot.process_new_messages([update.message])
    
    return func.HttpResponse(body='', status_code=200)
