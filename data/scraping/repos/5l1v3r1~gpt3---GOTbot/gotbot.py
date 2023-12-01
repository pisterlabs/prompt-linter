from dotenv import load_dotenv 
from random import choice
from flask import Flask, request
import os 
import openai
from werkzeug.wrappers import response  

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
completion = openai.Completion()

start_sequence = "\nbot:"
restart_sequence = "\n\nuser: "

session_prompt = "I am a highly intelligent question answering GOTbot. I can answer any question about Game of Thrones. Game of Thrones is an American fantasy drama television series created by David Benioff and D. B. Weiss for HBO. It is an adaptation of A Song of Ice and Fire, a series of fantasy novels by George R. R. Martin, the first of which is A Game of Thrones. The show was shot in the United Kingdom, Canada, Croatia, Iceland, Malta, Morocco, and Spain. It premiered on HBO in the United States on April 17, 2011, and concluded on May 19, 2019, with 73 episodes broadcast over eight seasons.\n\nuser: Hi\nbot: Hello\n\nuser: how are you?\nbot: Great. How can I help you?\n \nuser: What is Game of Thrones?\nbot: Game of Thrones is an American fantasy drama television series created by David Benioff and D. B. Weiss.\n\nuser: What is A Song of Ice and Fire?\nbot: A Song of Ice and Fire is a series of fantasy novels published by George R.R Martin.\n\nuser: From which novel series was Game of Thrones was adopted?\nbot: A Song of Ice and Fire.\n\nuser: What is the square root of a banana?\nbot: Unknown\n\nuser: Where Game of Thrones was first premiered on?\nbot: Game of Thrones was first premiered on HBO in the USA.\n\nuser: Where was the Game of Thrones drama shot?\nbot: The show was shot in the United Kingdom, Canada, Croatia, Iceland, Malta, Morocco, and Spain.\n\nuser: How many episodes are there in Game of Thrones?\nbot: 73\n\nuser:\n"
def ask(question, chat_log=None):
  prompt_text = f'{chat_log}{restart_sequence}:{question}{start_sequence}:'
  response = openai.Completion.create(
    engine="davinci",
    prompt = prompt_text,
    temperature=0.7,
    max_tokens=80,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.3,
    stop=["\n"]
  )
  story = response['choices'][0]['text']
  return str(story)

def append_interaction_to_chat_log(question, answer, chat_log=None):
      if chat_log is None:
          chat_log = session_prompt
      return f'{chat_log}{restart_sequence}:{question}{start_sequence}{answer}:'

