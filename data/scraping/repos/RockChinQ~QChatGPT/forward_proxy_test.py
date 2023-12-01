import os

import openai

client = openai.Client(
    api_key=os.environ["OPENAI_API_KEY"],
)

openai.proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
}

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?",
        }
    ]
)

print(resp)