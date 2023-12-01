from dotenv import load_dotenv
from random import choice
from flask import Flask, request
import os
import openai
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
completion = openai.Completion()

start_sequence = "\nAI:"
restart_sequence = "\n\nHuman:"
session_prompt = "The AI is a chatbot that is based on principles of nonviolent communication. The AI repeatedly asks if, with whom, what, where, when, how, and why a conflict is happening and then synthesizes observations,feelings, needs and requests for the human.\n\nHuman:"

def ask(question, chat_log=None):
 prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
 response = openai.Completion.create(
 engine="curie-instruct-beta",
 prompt=prompt_text,
 temperature=0.9,
 max_tokens=150,
 top_p=1,
 frequency_penalty=0.1,
 presence_penalty=0.9,
 stop=["\n"],
 )
 story = response['choices'][0]['text']
 return str(story)

def append_interaction_to_chat_log(question, answer, chat_log=None): 
 if chat_log is None:  
     chat_log = session_prompt 
 return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'
