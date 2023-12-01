import os

import openai
from db import JOURNAL_DELIMETER
from dotenv import load_dotenv


def load_ai():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def return_response(prompt, closest_docs):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"""Respond based on journal entries provided in the next message that are separated by {JOURNAL_DELIMETER}) 
                  Be prophetic and cite examples in your responses from the journals.""",
            },
            {"role": "assistant", "content": closest_docs},
            {"role": "user", "content": prompt},
        ],
    )

    return response["choices"][0]["message"]["content"]
