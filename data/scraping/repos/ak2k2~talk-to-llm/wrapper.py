# Property of JargonBots
# Written by Armaan Kapoor on 12-26-2022

import os
from dotenv import load_dotenv
import openai
from pathlib import Path

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenaiRequest:
    prompt = None
    tokensout = None

    def __init__(self, prompt):
        self.prompt = prompt

    def return_tokens(self, temperature, max_tokens, frequency_penalty, stop):
        self.tokensout = openai.Completion.create(
            model="text-davinci-003",
            prompt=self.prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=frequency_penalty,
            presence_penalty=0.0,
            stop=stop,
        )
        return self.tokensout["choices"][0]["text"].strip()
