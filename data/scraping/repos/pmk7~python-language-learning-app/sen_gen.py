import os
import openai
from data import *


class SentenceGenerator:
    """Generates a sentence based on a given word, fetched from the OpenAI API"""

    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def generate_sentence(self, word, max_tokens=50, temperature=0.9):
        prompt = f"Ein Beispiel mit: {word}"

        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=[".", "!", "?"]
            )

            response_text = response.get('choices')[0].get('text')
            return response_text.strip()
        except openai.error.OpenAIError as error:
            print(f"Error connecting to the OpenAI API: {error}")
            return None
        except Exception as error:
            print(f"Unexpected error: {error}")
            return None

    def test_api(self, word):
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt="Test: ",
                max_tokens=5,
                temperature=0.9,
                stop=[".", "!", "?"]
            )

            if response.status_code == 200:
                return True
            else:
                return False
        except openai.error.OpenAIError as error:

            # print(f"Error connecting to the OpenAI API: {error}")
            return False
        except Exception as error:
            # print(f"Unexpected error: {error}")
            return False
