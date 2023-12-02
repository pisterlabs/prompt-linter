import json
import os
import openai

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "Given the following sample input:\n    question: Where is Rome? \n    bot_answer: Star Wars is a movie.\n\nPlease validate the bot's answer and provide a binary JSON response containing either \"YES\" or \"NO\" using the following format:\n```json\n{{\n  \"validation\": \"NO\"\n}}\n```"
        },
        {
            "role": "user",
            "content": "question:Co jest stolicą Polski?\nanswer: Kraków"
        },
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

print(response)
print('-----------')
content = response.choices[0].message.content
print(content)
print('-----------')

validated_answer = json.loads(content)["validation"]
print(f'answer: {validated_answer}')
