import os

import openai
from dotenv import load_dotenv

load_dotenv()


def chat_with_gpt_3(question: str, max_tokens: int = 500) -> str:
    """
    Function to interact with GPT-3
    :param question: question to ask
    :param max_tokens: max tokens to generate
    :return: gpt-3 response
    """
    api_key: str = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    response = openai.Completion.create(engine="text-davinci-002", prompt=question, max_tokens=max_tokens)
    return response.choices[0].text
