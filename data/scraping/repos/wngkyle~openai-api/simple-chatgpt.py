import os
import openai 
from dotenv import load_dotenv

load_dotenv('../.env')

openai.api_key = os.getenv("OPENAI_API_KEY")
chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{
        "role": "user", 
        "content": "Hi"
    }]
)
print(chat_completion.choices[0].message.content)
