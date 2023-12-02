# Set up your OpenAI API credentials
# import openai
# openai.api_key = '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'

# # Define the input prompt
# prompt = 'act as a sassy girl'
# user_input = ''

# # Generate a response using ChatGPT
# def generate_response(prompt, user_input):
#     response = openai.Completion.create(
#         model='text-davinci-003',
#         # engine='davinci-codex',
#         prompt=prompt + ' ' + user_input,
#         temperature=0.7,
#         max_tokens=100,
#         n=1,
#         stop=None,
#         # temperature_scaling=False,
#         top_p=None,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     return response.choices[0].text.strip()

# # Continuously prompt for user input and get responses
# while True:
#     user_input = input('User: ')
#     prompt += '\nUser: ' + user_input
#     response = generate_response(prompt, user_input)
#     prompt += '\nAI: ' + response
#     print('AI:', response)
    

import openai
import pyttsx3
import os

openai.api_key = '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'

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Define the input prompt
prompt = 'act as Princess'
user_input = ''

# Generate a response using ChatGPT
def generate_response(prompt, user_input):
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt + ' ' + user_input,
        temperature=0.7,
        max_tokens=256,
        n=1,
        stop=None,
        top_p=None,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

# Continuously prompt for user input and get responses
while True:
    user_input = input('User: ')
    prompt += '\nUser: ' + user_input
    response = generate_response(prompt, user_input)
    prompt += '\n' + response
    print('AI:', response)
    # responses=response.strip("Princess")
    with open ('girlai.txt','a', encoding='utf-8') as f:
        e=response
        f.writelines(e)
        f.writelines("\n")

    # Use text-to-speech to speak the AI's response
    engine.say(response)
    engine.runAndWait()