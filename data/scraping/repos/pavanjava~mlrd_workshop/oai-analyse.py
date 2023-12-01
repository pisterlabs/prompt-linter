# This code is for v1 of the openai package: pypi.org/project/openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os


_ = load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "You will be provided with a sentence in English, and your task is to predict the sentiment of the given sentense as POSITIVE, NEGATIVE, NEUTRAL and also their sentiment score which is always positive in the format {sentiment: POSITIVE, score: 0.00}"
        },
        {
            "role": "user",
            "content": "i saw a movie last week which is worst and waste of time"
        }
    ],
    temperature=0,
    max_tokens=256
)

print(response)
