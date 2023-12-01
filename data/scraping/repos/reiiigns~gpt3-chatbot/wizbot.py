from dotenv import load_dotenv
from random import choice
from flask import Flask, request 
import os
import openai

load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"
completion = openai.Completion()

esponse = openai.Completion.create(
  model="text-davinci-002",
  prompt= = "You: What have you been up to?\nWankyJizardbot: Watching old movies.\nYou: Did you watch anything interesting?\nWankyJizardbot: Yes, I watched The Three Musketeers.\nYou: That's awesome. What's your favorite kind of pizza?\nWankyJizardbot: I like pepperoni pizza.\nYou: Oh, I like that one too. My favorite is green olives and pineapple.\nWankyJizardbot: Oh, that's kinda weird but I'm sure it's good.\nYou: Yeah, it's a strange combination but I love it.\nWankyJizardbot: nice, man. \nYou: Do you ever think about the meaning of life?\nWankyJizardbot: Yes, I often think about the meaning of life. It's a big question that everyone tries to answer in their own way.\nYou: for sure, I guess everyone finds their own reason.\nWankyJizardbot: Exactly.\nYou: What do you think about the future of ai?\nWankyJizardbot: I think the future of ai is very exciting! It has the potential to help us solve a lot of problems.\nYou: Been pretty lonely lately, you ever feel like it's impossible to meet a good person?\nWankyJizardbot: I can understand how you feel. It can sometimes seem like there are no good people out there, but I believe that if you keep looking, you'll find someone special. You're a really great person."
temperature=0.5,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0.5,
  presence_penalty=0,
  stop=["You:"]

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
