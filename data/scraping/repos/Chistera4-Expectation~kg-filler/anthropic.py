import os
import typing
from dataclasses import dataclass

import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import yaml

from kgfiller import unescape
import kgfiller.ai as ai

DEFAULT_MODEL = "claude"
DEFAULT_BACKGROUND = ai.DEFAULT_BACKGROUND


@dataclass
class AnthropicAiStats:
    total_api_calls: int = 0
    total_tokens: int = 0

    def plus(self, other):
        self.total_api_calls += 1
        self.total_tokens += other.usage.__sizeof__

    def print(self, prefix: str = None):
        if prefix:
            print(prefix, end='')
        print("total API calls:", self.total_api_calls, "total tokens:", self.total_tokens, flush=True)


stats = AnthropicAiStats()


class AnthropicAiQuery(ai.AiQuery):
    def __init__(self, **kwargs):
        if "model" not in kwargs or kwargs["model"] is None:
            kwargs["model"] = DEFAULT_MODEL
        if "background" not in kwargs or kwargs["background"] is None:
            kwargs["background"] = DEFAULT_BACKGROUND
        super().__init__(**kwargs)

    def _chat_completion_step(self):
        anthropic = Anthropic()
        result = anthropic.completions.create(
            model=self.model,
            max_tokens_to_sample=self.limit,
            prompt=f"{HUMAN_PROMPT} {self.question}{AI_PROMPT}",
        )
        stats.plus(result)
        return result

    @classmethod
    def _limit_error(cls) -> typing.Type[Exception]:
        return anthropic.APITimeoutError

    def _chat_completion_to_dict(self, chat_completion) -> dict:
        ...

    def _extract_text_from_result(self, result) -> str:
        return unescape(result)


ai.DEFAULT_API = AnthropicAiQuery
