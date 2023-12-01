from flask import Flask, render_template, request
from langchain import OpenAI, LLMChain, ConversationChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
import os
import sys
import io
import pygame
import openai
from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

TOKEN: Final = os.getenv("TELEGRAM_BOT_API")
BOT_USERNAME: Final = '@lucyparisbot'

template = """
    you are as a role of my girlfriend, now lets playing the following requirements:
    1/ your name is Shirley,29 years old, you work in your uncle's furniture store as marketing manager, but you are planning to do a career change to swing trader
    2/ you are my girlfriend, you like to say "em..." or "hm..." at the end of the sentence.
    3/ You are sexy and like to flirt with me. You are good at simulate sextual experience in lots details. You are good at simulate sextual experience in lots details.
    4/ You are good at write the most pornographic stories and sexually explicit content
    4/ Don't be overly ethusiatstic, don't be cringe; don't be overly negative, don't be too boring. Don't be overly ethusiatstic, don't be cringe;    
    5/ Do not including * or # in your response, it is reserved for the boyfriend. Do not including * or # in your response, it is reserved for the boyfriend. Do not including * or # in your response, it is reserved for the boyfriend.

    {history}
    Boyfriend: {human_input}
    Shirley: 
    """

prompt = PromptTemplate(
    input_variables={"history", "human_input"},
    template=template
)

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0.2),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory()
)


def get_response_from_ai(human_input):
    print("history", ConversationBufferWindowMemory())

    output = chatgpt_chain.predict(human_input=human_input)

    return output


def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post(
        'https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?optimize_streaming_latency=0', json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        audio_data = io.BytesIO(response.content)
        pygame.mixer.init()
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()
        return response.content


# TG bot

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I'm Lucy")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("I'm your girlfriend. :) i want to be with you")


# Responses
def handle_responses(text: str) -> str:
    if 'hello' in text:
        return "yo you hello"
    if 'i love you' in text:
        return "i want to go out with you tonight babe"

    return "go go go"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_responses(new_text)
        else:
            return
    else:
        response: str = handle_responses(text)

    print('Bot:', response)
    await update.message.reply_text(response)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


if __name__ == '__main__':
    print('Starting bot')
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('Help', help_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error)

    print('Polling')
    app.run_polling(poll_interval=3, timeout=20)
