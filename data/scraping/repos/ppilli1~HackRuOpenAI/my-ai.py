import openai
from dotenv import load_dotenv
import os


#load_dotenv("API_KEY.env")
API_KEY = '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'
openai.api_key = API_KEY


response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the FIFA 2014 World Cup?"}
    ]
)
print(response['choices'][0]['message']['content'])