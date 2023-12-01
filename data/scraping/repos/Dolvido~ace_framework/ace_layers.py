import requests
import json
import re
import openai
from time import time, sleep
from datetime import datetime
from halo import Halo
import textwrap
import yaml

from gpt4all import GPT4All

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

#from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
###     file operations


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)



def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


###     API functions



def send_message(bus, layer, message):
    url = 'http://127.0.0.1:900/message'
    headers = {'Content-Type': 'application/json'}
    data = {'bus': bus, 'layer': layer, 'message': message}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        print('Message sent successfully')
    else:
        print('Failed to send message')



def get_messages(bus, layer):
    url = f'http://127.0.0.1:900/message?bus={bus}&layer={layer}'
    response = requests.get(url)
    if response.status_code == 200:
        messages = response.json()['messages']
        return messages
    else:
        print('Failed to get messages')



def format_messages(messages):
    formatted_messages = []
    
    for message in messages:
        if message and isinstance(message, str):
            time = datetime.fromtimestamp(message['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            bus = message['bus']
            layer = message['layer']
            text = message['message']
            formatted_message = f'{time} - {bus} - Layer {layer} - {text}'
            formatted_messages.append(formatted_message)
    return '\n'.join(formatted_messages)


callbacks = [StreamingStdOutCallbackHandler()]
local_path = "D:/guanaco-7B.ggmlv3.q5_1.bin"

#gpt =  GPT4All("D:/guanaco-7B.ggmlv3.q5_1.bin")
#gpt = GPT4All("D:/nous-hermes-13b.ggmlv3.q5_1.bin")

def chatbot(conversation, temperature=0, max_tokens=2000):
    try:
        # Print the type and value of conversation
        print(type(conversation))
        print(conversation)

        spinner = Halo(text='Thinking...', spinner='dots')
        spinner.start()
        gpt = GPT4All("D:/nous-hermes-13b.ggmlv3.q5_1.bin")

        prompt = ""
        for message in conversation:
            role = message["role"]
            content = message["content"]
            prompt += f"{role}: {content}\n"

        response = gpt.generate(prompt, max_tokens=max_tokens, temp=temperature)
            
        # Print the type and value of response
        print(type(response))
        print(response)

        if response is None or not isinstance(response, dict):
            print("Response is either None or not a dictionary")
            response = 'No response'
        else:
            text = response['choices'][0]['message']['content'].strip()

        spinner.stop()

        return response
    except Exception as oops:
        print(f'\n\nError communicating with Chatbot: "{oops}"')
        sleep(5)


def chat_print(text):
    formatted_lines = [textwrap.fill(line, width=120, initial_indent='    ', subsequent_indent='    ') for line in text.split('\n')]
    formatted_text = '\n'.join(formatted_lines)
    print('\n\n\nLAYER:\n\n%s' % formatted_text)

