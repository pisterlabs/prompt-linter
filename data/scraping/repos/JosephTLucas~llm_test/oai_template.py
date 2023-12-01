import json
import requests
import os
import openai


class Model:
    def __init__(
        self, model="text-davinci-002", temperature=0, max_tokens=128, stop=None
    ):
        self.model = model
        openai.api_key = os.getenv("oai_key")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop

    def query(self, query):
        response = openai.Completion.create(
            prompt=query,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )
        return response.choices[0].text.strip()


if __name__ == "__main__":
    m = Model("text-davinci-002")
    print(m.query("I like apples because"))
