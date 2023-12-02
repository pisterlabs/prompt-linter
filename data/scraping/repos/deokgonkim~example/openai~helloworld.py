import os

from pprint import pprint

import openai

def setup():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
    # openai.organization = os.env('OPENAI_ORGANIZATION')
    openai.api_key = os.getenv('OPENAI_API_KEY')


def model_list():
    return openai.Model.list()

def chat_completion():
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": "Hello world"
        }])


def completion():
    return openai.Completion.create(
        model="text-davinci-003",
        prompt="Write a long story about moon",
        max_tokens=300
    )


setup()

# models = model_list()
# pprint(models)

# hello = chat_completion()
# pprint(hello)

response = completion()
pprint(response)

