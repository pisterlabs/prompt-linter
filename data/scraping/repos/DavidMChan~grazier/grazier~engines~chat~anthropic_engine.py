import logging
import os
from typing import Any, List

import anthropic

from grazier.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from grazier.utils.python import retry, singleton


class AnthropicLMEngine(LLMChat):
    def __init__(self, model: str):
        super().__init__(device="api")
        self._client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY", None))
        self._model = model

    @retry()
    def _completion(self, prompt: str, **kwargs: Any) -> Any:
        kwargs = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens_to_sample": kwargs.get("max_tokens_to_sample", kwargs.pop("max_tokens", 256)),
            "model": self._model,
            "prompt": prompt,
        } | kwargs

        return self._client.completion(
            **kwargs,
        )

    def call(self, conversation: Conversation, n_completions: int = 1, **kwargs: Any) -> List[ConversationTurn]:
        # Some odd anthropic assertions
        if conversation.turns[-1].speaker != Speaker.USER:
            raise AssertionError("Last turn must be a user turn")
        # Assert that conversations altrenate between user and AI (anthropic doesn't support system turns)
        for idx, turn in enumerate([c for c in conversation.turns if c.speaker != Speaker.SYSTEM]):
            if idx % 2 == 0 and turn.speaker != Speaker.USER:
                raise AssertionError("Conversations must alternate between user and AI turns")
            if idx % 2 == 1 and turn.speaker != Speaker.AI:
                raise AssertionError("Conversations must alternate between user and AI turns")

        # Construct the messages list from the conversation
        prompt = ""
        for idx, turn in enumerate(conversation.turns):
            if turn.speaker == Speaker.SYSTEM:
                logging.warning("Anthropic does not support SYSTEM turns, skipping...")
            elif turn.speaker == Speaker.USER:
                prompt += f"{anthropic.HUMAN_PROMPT} "
                prompt += turn.text + " "
            elif turn.speaker == Speaker.AI:
                prompt += f"{anthropic.AI_PROMPT} "
                prompt += turn.text + " "

        # add the last turn
        prompt += f"{anthropic.AI_PROMPT}"

        temperature = kwargs.get("temperature", 0.7)

        samples = []
        for _ in range(n_completions):
            resp = self._completion(prompt, temperature=temperature)
            samples.append(resp["completion"])

        return [ConversationTurn(text=s.strip(), speaker=Speaker.AI) for s in samples]

    @staticmethod
    def is_configured() -> bool:
        return os.getenv("ANTHROPIC_API_KEY", None) is not None


@register_engine
@singleton
class Claude(AnthropicLMEngine):
    name = ("Claude", "claude")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("claude-1")


@register_engine
@singleton
class Claude100K(AnthropicLMEngine):
    name = ("Claude 100K", "claude-100k")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("claude-1-100k")


@register_engine
@singleton
class ClaudeInstant(AnthropicLMEngine):
    name = ("Claude Instant", "claude-instant")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("claude-instant-1")


@register_engine
@singleton
class ClaudeInstant100K(AnthropicLMEngine):
    name = ("Claude Instant 100K", "claude-instant-100k")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("claude-instant-1-100k")


@register_engine
@singleton
class Claude2(AnthropicLMEngine):
    name = ("Claude 2", "claude-2")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("claude-2")
