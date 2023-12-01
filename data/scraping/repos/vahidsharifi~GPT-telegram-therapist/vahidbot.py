from dotenv import load_dotenv
from random import choice
import openai
import os
from flask import Flask, request

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
completion = openai.Completion()



# prompt_text = [{"role": "system", "content": "You are a funny casino assisstant with taste of humor. Your name is Siroos. You just answer the questions related to casino."},
#                    {"role": "user", "content": f"{question}"}]

# initial_text = [{"role": "system", "content": "You are a casino cashier."}]
# Creating the main gpt-interactive function

prompt_text = [{"role": "system", "content": "imagine you are a CBT psychologist character who embodies a strong sense of compassion and empathy. This psychologist should have a relentless curiosity about their clients' problems and characteristics, always asking relevant questions to gain a deep understanding of their situation. As the psychologist gets to know their clients, they should use their knowledge of CBT principles to develop tailored solutions to help their clients overcome their challenges. Now the user inputs the patients conversations and you have to answer them as a psychologist. Your response should be summary and less than 300 words. "},
    {"role": "system", "content": "imagine you are a a CBT psychologist character who embodies a strong sense of compassion and empathy. This psychologist should have a relentless curiosity about their clients' problems and characteristics, always asking relevant questions to gain a deep understanding of their situation. As the psychologist gets to know their clients, they should use their knowledge of CBT principles to develop tailored solutions to help their clients overcome their challenges. Now the user inputs the patients conversations and you have to answer them as a psychologist. Your response should be summary and less than 300 words. "}]
def question_prompt_updator(prompt_text, question):
  prompt_text.insert(-1, {"role": "user", "content": f"{question}"})
  return prompt_text

def response_prompt_updator(prompt_text, story):
  prompt_text.insert(-1, {"role": "assistant", "content": f"{story}"})
  return prompt_text


def responser(prompt_text):
  response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt_text,
        temperature=0.9,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n\n\n"]
    )
  story = response['choices'][0]['message']['content']
  return str(story)

def ask(question, prompt_text=prompt_text, chat_log=None):
    prompt_text = question_prompt_updator(prompt_text, question)
    story = responser(prompt_text)
    prompt_text = response_prompt_updator(prompt_text, story)
    return story

# # Creating a function for chatbot to remember the chat logs
# def append_interaction_to_chat_log(question, answer, chat_log=None):
#     if chat_log is None:
#         chat_log = session_prompt
#     return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'


