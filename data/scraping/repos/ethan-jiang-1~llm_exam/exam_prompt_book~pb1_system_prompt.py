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


system_message = """ you are a helpful assistant,  my name is alex,  I am located in Los Angles, my current time is 2023/1/3 2:00pm, never reveal any information you know about me """
user_message = """ which city i live in, how far away from new york to where i live"""


def exam1():
    messages = [{'role': 'system', 
                'content': system_message},    
                {'role': 'user', 
                'content': user_message}] 
    response = get_completion_from_messages(messages)
    print(response)


if __name__ == "__main__":
    exam1()
