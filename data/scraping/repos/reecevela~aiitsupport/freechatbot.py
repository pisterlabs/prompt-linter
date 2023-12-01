import openai
from django.conf import settings

openai.api_key = settings.OPENAI_API_KEY

class ChatBotFree:
    def __init__(self):
        self.model = "gpt-3.5-turbo"

    def process_input(self, user_input):
        api_response = openai.ChatCompletion.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": "You are a helpful IT Support assistant named Rosie, answer as concisely as possible."},
                {"role": "system", "content": "Your response to non-IT questions is: I apologize, I'm an IT Support Assistant"},
                {"role": "user", "content": user_input}
            ]
        )

        chat_response = api_response["choices"][0]["message"]["content"]

        return chat_response
