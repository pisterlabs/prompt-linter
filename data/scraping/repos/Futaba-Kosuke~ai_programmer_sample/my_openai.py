from dataclasses import InitVar, dataclass, field
from typing import Any

import openai

from abstracts import AbstractGenerator


@dataclass
class OpenAiGenerator(AbstractGenerator):
    api_key: InitVar[str] = None
    completion: Any = field(init=False)

    def __post_init__(self, api_key: str) -> None:
        openai.api_key = api_key
        self.completion = openai.Completion

    def generate(self, prompt: str) -> Any:
        return self.completion.create(
            engine="text-davinci-002", prompt=prompt, max_tokens=256
        )
