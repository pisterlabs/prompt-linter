import os
import openai
from config import apikey

openai.api_key = apikey


def chat(query):
    try:
        response = openai.Completion.create(
          model="text-davinci-003",
          prompt=query,
          temperature=1,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        return (response['choices'][0]['text']).strip()
    except Exception as e:
        return ""
