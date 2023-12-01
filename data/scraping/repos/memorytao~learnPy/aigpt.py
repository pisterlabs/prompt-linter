import openai
import os


API_KEY = "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"
openai.api_key = API_KEY
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "how to deploy uvicorn"}
    ]
)

print(completion.choices[0].message.content)
