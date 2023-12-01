from dotenv import load_dotenv
from random import choice
from flask import Flask, request, session
import os
import openai

load_dotenv()
#openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = "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"
completion = openai.Completion()

start_sequence = "\nSyndrum:"
restart_sequence = "\n\nPerson"
session_prompt = "You are talking with Syndrum, GPT3 bot influencer who was very inteligent about new technology and programing. Syndrum owned by Sayuru. He is a software engineer.You can ask anything from syndrum"

def ask(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt_text,
      temperature=0.8,
      max_tokens=150,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.3,
      stop=["\n"],
    )
    story = response['choices'][0]['text']
    return str(story)

def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = session_prompt
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'