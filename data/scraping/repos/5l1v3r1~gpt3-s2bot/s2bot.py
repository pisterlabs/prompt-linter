from dotenv import load_dotenv
from random import choice
from flask import Flask, request
import os
import openai

load_dotenv()
openai.api_key=os.getenv('OPENAI_API_KEY')
completion = openai.Completion()

start_sequence="\nAlfredo:",
restart_sequence="\n\nPerson:"
session_prompt="You are talking to Alfredo, GPT3 bot influencer who was mentored by Elon Musk",

def ask(question, chat_log=None):
    prompt_text = f"{chat_log} {restart_sequence}: {question}{start_sequence}"
    response=openai.Completion.create(
        engine="davinci",
        prompt=prompt_text,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n"]
    )
    story=response['choices'][0]['text']
    return str(story)

def append_interation_to_chat_log(quotation, answer, chat_log=None):
    if chat_log is None:
        chat_log=session_prompt
    return f"{chat_log} {restart_sequence} {quotation} {start_sequence} {answer}"



