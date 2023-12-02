import requests
import json
import openai
import os

class BaseLLM1():

    __API_KEY = os.getenv('OPENAI_API_KEY')
    __API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def _get_headers(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BaseLLM1.__API_KEY}",
        }
        return headers

    def _get_data(self, messages, model="gpt-3.5-turbo", temperature=1):
        data = {
            "model": model,
            "messages": [{"role": "user", "content": messages}],
            "temperature": temperature,
            "max_tokens": 1000
        }
        return data

    def _get_response(self, headers, data):
        response = requests.post(BaseLLM1.__API_ENDPOINT, headers=headers, data=json.dumps(data))
        return response
