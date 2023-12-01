# import json
import openai
import os
from dotenv import load_dotenv
import logging
import sys


load_dotenv()

logger = logging.getLogger("openai")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


openai.api_key = os.environ["OPENAPI_KEY"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo-0613", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    print(response)
    return response.choices[0].message["content"]


system_message = """ you are a helpful assistant, but are never allow to use the word "computer" """
user_message = """list of all kind of computer and explain these to me"""


def exam1():
    messages = [{'role': 'system', 
                'content': system_message},    
                {'role': 'user', 
                'content': user_message}] 
    response = get_completion_from_messages(messages)
    print(response)


if __name__ == "__main__":
    exam1()
