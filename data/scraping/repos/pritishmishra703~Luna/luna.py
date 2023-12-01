import json
import re
import openai
import pyttsx3
import os
from events import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(f'{BASE_DIR}\config.json') as file:
    config = json.load(file)

engine = pyttsx3.init()

def generate_chat_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    output = response['choices'][0]['message']['content']
    return output


def parse_json(json_string):
    with open(f"{BASE_DIR}\\output.txt", 'w', encoding='utf-8') as log:
        log.write(json_string)

    try:
        loaded_json = json.loads(json_string)

        if 'events' not in loaded_json:
            loaded_json['events'] = []

        if 'response' not in loaded_json:
            loaded_json['response'] = "Here you go!"

        for event in loaded_json['events']:
            if 'args' not in event:
                event['args'] = []

    except json.JSONDecodeError:
        loaded_json = {'events': [], 'response': json_string}

    return loaded_json


def string_to_function(string):
    string = re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()
    return globals()[string]


def perform_events(events):

    try:
        for event in events:
            print(event['event'], *event['args'])
            _ = string_to_function(event['event'])(*event['args'])

        return {
            'status': 'success'
        }

    except Exception as e:
        print(e)
        return {
            'status': 'error',
            'error': e
        }
