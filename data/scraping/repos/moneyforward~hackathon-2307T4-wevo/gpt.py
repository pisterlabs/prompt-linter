import os
import openai

openai.organization = os.getenv('OPENAI_ORG_TOKEN')
openai.api_key = os.getenv('OPENAI_API_KEY')


def chat_with_gpt3(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
    )
    return response.choices[0].message


def evaluate_with_gpt3(messages, functions):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        functions=functions,
        function_call={"name": "insert_evaluation"},
    )
    return response.choices[0].message
