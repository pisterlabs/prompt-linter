import requests
from dotenv import load_dotenv
import os
import requests
import json
from openai import OpenAI


# Load the API key for Perplexity from .env file

load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


import requests

url = "https://api.perplexity.ai/chat/completions"

payload = {
    "model": "mistral-7b-instruct",
    "messages": [
        {
            "role": "system",
            "content": "Be precise and concise."
        },
        {
            "role": "user",
            "content": "What is the latest publication about the periodonatl disease classification?"
        }
    ],
    "max_tokens": 1000,
    "temperature": 0.7
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer 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"
}

response = requests.post(url, json=payload, headers=headers)



# Formating the answer in a more readable way
answer = json.loads(response.text)
answer = answer['choices'][0]['message']['content']
print(answer)