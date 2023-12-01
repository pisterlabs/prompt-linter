import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


# AlfredChat.py
class AlfredChat:
    def __init__(self, api_key):
        self.api_key = api_key

    # @staticmethod
    def return_completion(self, prompt: str):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user", "content": prompt
                }
            ]
        )

        return response['choices'][0]['message']['content']


