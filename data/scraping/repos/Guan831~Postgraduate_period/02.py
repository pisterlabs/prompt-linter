# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
from dotenv import load_dotenv
import openai
import os
OPENAI_API_KEY = "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"


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    pass


if __name__ == "__main__":
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Today I went to the movies and...",
        temperature=1,
        max_tokens=60,
    )

print(response)
