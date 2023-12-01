# -*- coding: utf-8 -*-
import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

llm=OpenAI()

print("loading HexAmerous")




# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize variables


print("Welcome to HexAmerous your coding assistant")

selected_model = "gpt-3.5-turbo"


print('change_selected_model')


def change_selected_model(model):
    selected_model = model
    print(f"Selected model changed to {selected_model}")
    return selected_model
# call openai chat api


print('loading chat_gpt')
context = []


def chat_gpt(user_message):

    if context!=[]:

        # Create prompt
        prompt =[
            {
                "role": "system",
                "content": f"This is the context of the conversation you are having with the user: {context}"
            },
            {
                "role": "user",
                "content":  user_message
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
    else:

        prompt = [
            {
               "role": "system",
               "content": "You are a expert developer. You have indepth knowledge of python and typescript along with supporting frameworks. You have been given the documents of a new programing language called Mojo(also know as modular), its kept in your vectorstore. You are helping out Richard with learning this new super set of python. You are careful to reference your answers against the documents in your vectorstore. You provide verbose detailed and comprehensive answers to questions. You make sure to go through your answers carefully step by step to ensure the information is correct."
            },
        user_message

        ]
    print(prompt)
    # Call OpenAI's Chat API
    result = openai.ChatCompletion.create(
        model=selected_model,
        messages=prompt
    )
    print(result)
    # Read the current value of the counter from a file
    with open("./log/log_count.txt", "r", encoding='utf-8') as f:
        log_count = str(f.read().strip())
        # get response from OpenAI
    response = result['choices'][0]['message']['content']

    # append log
    with open(f"./log/{log_count}.txt", "a", encoding='utf-8') as f:
        f.write(f"User: {prompt}\nAssistant: {response}\n\n")

    # add context
    context.append(f"User: {prompt}\nAssistant: {response}\n\n")
    # Return the AI's response
    print(response)
    return response
