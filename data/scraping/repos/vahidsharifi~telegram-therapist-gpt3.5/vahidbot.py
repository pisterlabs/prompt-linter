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
def ask(question, chat_log=None):
    prompt_text = [{"role": "system", "content": "You are a master degree student at univesity and you want to write an academic paragraphs about what you are asked for."},
                   {"role": "user", "content": f"{question}"}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt_text,
        temperature=0.9,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n"]
    )
    story = response['choices'][0]['message']['content']
    return str(story)


# # Creating a function for chatbot to remember the chat logs
# def append_interaction_to_chat_log(question, answer, chat_log=None):
#     if chat_log is None:
#         chat_log = session_prompt
#     return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'

