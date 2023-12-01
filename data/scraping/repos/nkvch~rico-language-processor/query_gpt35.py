import openai
import os

openai.api_key = os.environ['OPENAI_API_KEY']

def query_openai_gpt35(messages, max_tokens=1000):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=max_tokens
    )

    return completion.choices[0].message.content
