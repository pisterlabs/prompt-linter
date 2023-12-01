from abc import ABC, abstractmethod
from typing import Any
import openai

class BotDalle(ABC):
   
    def __init__(self, settings: Any):
        pass


    def get_bot_response(self, text: str) -> str:
        prompt = text
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="256x256",
        )
        return response["data"][0]["url"]
