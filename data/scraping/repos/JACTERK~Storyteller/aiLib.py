# Made with <3 by Jacob Terkuc

import os
from collections import deque
import openai
import settings
import time
import asyncio
from dotenv import load_dotenv
from datetime import datetime

# Load Discord and OpenAI API keys
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")


<<<<<<< HEAD
# Function to generate response
def determine_Response(prompt):
    # Debug
    if settings.debug_mode:
        print(prompt)

    # Generate Response using Chat API
    response = openai.ChatCompletion.create(model=settings.model_gen, messages=prompt)

    # Return response
    return response
=======
translate_dict = {"'": "", '"': "", "’": ""}


# Function to generate response, takes a library 'msg' and calls the OpenAI API to generate a response. Returns a
# response as a string.
def generate(msg, return_type=str, model='gpt-4'):
    if settings.debug:
        print("Function generate() called | Return type is : " + str(return_type) + " | Model is: " + model)
    msglist = [{"role": "system", "content": str(msg)}]

    # Checks if the type of 'msg' is a list or a string
    # In this case, the input is assumed to be a conversation.
    if type(msg) == list:
        msglist = msg

    # TODO: Hacky way to retry if the API fails. Fix this in the future.
    try:
        # Create a new chatcompletion using the OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=msglist
        )
    except TypeError as e:
        # Create a new chatcompletion using the OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=msglist
        )

    if settings.debug:
        print(response)

    response["choices"][0]['message']['content'] = response["choices"][0]['message']['content']\
        .translate(translate_dict)

    # Determine how the response should be returned (Either as a list or a string. By default, is returned as a string)
    if return_type == list or return_type == deque:
        return response
    elif return_type == str:
        return response["choices"][0]['message']['content']
    else:
        print("Error: Invalid return type (Valid types: list, deque, string). Returning string...")
        return response["choices"][0]['message']['content']
>>>>>>> remove_userlist_py


# TODO: Implement this feature in the future
# Function to generate an image from a prompt
def generate_image(imagedict):
    # Generate image
    try:
        url = openai.Image.create(prompt=imagedict["prompt"], n=1, size="1024x1024")

        imagedict["url"] = url["data"][0]["url"]

        if settings.debug:
            print("Image generated: " + url)

        imagedict["pass"] = True

    except Exception as e:
        print(f"Image Gen Failed: {e}")
        imagedict["pass"] = False

    return imagedict


def log_gen(log_data):
    file_name = 'logfile.log'

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_data = f'{current_time} - {log_data}'

    with open(file_name, 'a') as log_file:
        log_file.write(formatted_data + '\n')
    return


def generate_typetime(message):
    return len(message) / settings.t_speed_multiplier


# Takes two character objects 's' (self) and 'c' (character) and combines their chat logs into a single chat log.
# The number of loops is based on the length of the first passed in character's chat log. Empty chat log idx's are
# filled with empty chat log entries. Returns a deque of chat log entries.
def combine_chat_log(s, c, return_type=deque):
    combined_chat_log = deque([])

    for i in range(len(s.get_chat_log())):
        # Append 's' chat at index 'i' to combined_chat_log
        try:
            combined_chat_log.append(s.get_chat_log()[i])
        except IndexError:
            combined_chat_log.append({"role": "system", "content": ""})

        # Append 's' chat at index 'i' to combined_chat_log
        try:
            combined_chat_log.append(c.get_chat_log()[i])
        except IndexError:
            combined_chat_log.append({"role": "system", "content": ""})

    # Return the combined chat log as a deque by default, or as a string or list if specified
    if return_type == deque or return_type == str or return_type == list:
        return return_type(combined_chat_log)
    else:
        raise TypeError("Invalid return type (Must be deque or str)")
