from dotenv import load_dotenv
from random import choice
from flask import Flask, request

import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
completion = openai.Completion()

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "
session_prompt =  "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Great what is your favourite type of music\nAI: I like all types of music.\nHuman: my favourite is jazz tell me about jazz\nAI: Jazz is a type of music that originated in the United States in the late 19th and early 20th centuries. It is characterized by a complex structure and improvisation.\nHuman: name a jazz artist\nAI: Charles Mingus is one of my favorite jazz artists.\nHuman: how do I meditate\nAI: Meditation is an opportunity to turn inward and focus on the present moment. It is a time to quiet the mind and allow yourself to be at peace.\nHuman: how long should I meditate for\nAI: The length of time you meditate is up to you. Some people meditate for a few minutes, while others meditate for hours.\nHuman: is exercise good for the mind\nAI: Yes, exercise is good for the mind. It can help to improve your mood and memory.\nHuman: ",

def ask(question, chat_log=None):
  prompt_text = f'{chat_log} {restart_sequence}: {question} {start_sequence}:'
  response = openai.Completion.create(
  engine="text-davinci-002",
  temperature=0.9,
  max_tokens=150,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0.6,
  stop=[" Human:", " AI:"]  
  )
  story = response['choices'][0]['text']
  return str(story)

def append_interaction_to_chat_log(question, answer, chat_log=None):
  if chat_log is None: chat_log = session_prompt 
  return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'