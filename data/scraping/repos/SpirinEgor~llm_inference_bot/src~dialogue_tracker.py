from dataclasses import dataclass, field
from enum import Enum
from os import getenv
from time import time
from typing import Any

import openai
from loguru import logger


class MessageType(Enum):
    USER = "user"
    MODEL = "assistant"


@dataclass
class Dialogue:
    """Dataclass for storing dialogue history.
    History is a list of strings, where first string is the user input, second is the model response, etc.
    """

    user_id: str
    total_tokens: int = 0
    tokens: list[int] = field(default_factory=list)
    history: list[tuple[MessageType, str]] = field(default_factory=list)
    timestamp: float = field(default_factory=time)

    def pop(self):
        self.history.pop(0)
        tokens_removed = self.tokens.pop(0)
        self.total_tokens -= tokens_removed

    def update(self, user_message: str, model_message: str, prompt_tokens: int, completion_tokens: int):
        self.history.append((MessageType.USER, user_message))
        self.history.append((MessageType.MODEL, model_message))

        self.tokens.append(prompt_tokens - self.total_tokens)
        self.tokens.append(completion_tokens)

        self.total_tokens = prompt_tokens + completion_tokens
        self.timestamp = time()


class DialogueTracker:
    _OPENAI_API_KEY = "OPENAI_API_KEY"
    _MODEL_NAME = "gpt-4-1106-preview"
    _MODEL_CONTEXT_SAFE_SIZE = 128_000
    DEFAULT_ROLE = "You are a helpful assistant who always response in russian language."

    TOP_P = 0.9

    def __init__(self, seconds_to_reset: float = 60 * 60, messages_in_history: int = None):
        logger.info(
            f"Initializing ChatGPT based on '{self._MODEL_NAME}' model and nucleus sampling {self.TOP_P}. "
            f"Seconds to clear history: {seconds_to_reset}, max messages per history: {messages_in_history}"
        )
        openai_api_key = getenv(self._OPENAI_API_KEY)
        self._client = openai.AsyncOpenAI(api_key=openai_api_key)

        self._dialogue_history: dict[str, Dialogue] = {}
        self.max_alive_dialogue = seconds_to_reset
        self.messages_in_history = messages_in_history

        self._custom_roles: dict[str, str] = {}

    @property
    def config(self) -> dict[str, Any]:
        return {
            "messages_in_history": self.messages_in_history,
            "max_alive_dialogue": self.max_alive_dialogue,
        }

    def _validate_user_dialogue(self, user_id: str) -> bool:
        if user_id not in self._dialogue_history:
            return False

        dialogue = self._dialogue_history[user_id]

        current_time = time()
        if current_time - dialogue.timestamp > self.max_alive_dialogue:
            return False

        if len(dialogue.history) > 0 and dialogue.history[-1][0] is MessageType.USER:
            return False

        return True

    def _build_completion(self, user_message: str, user_id: str) -> list[dict]:
        if not self._validate_user_dialogue(user_id):
            if user_id in self._dialogue_history:
                del self._dialogue_history[user_id]
            self._dialogue_history[user_id] = Dialogue(user_id)

        dialogue = self._dialogue_history[user_id]

        while dialogue.total_tokens > self._MODEL_CONTEXT_SAFE_SIZE:
            dialogue.pop()

        if self.messages_in_history is not None:
            while len(dialogue.history) > self.messages_in_history:
                dialogue.pop()

        role = self.get_role(user_id)
        messages = [{"role": "system", "content": role}]
        for message_type, message in dialogue.history:
            messages.append({"role": message_type.value, "content": message})
        messages.append({"role": MessageType.USER.value, "content": user_message})

        return messages

    async def on_message(self, user_message: str, user_id: str) -> tuple[str, int]:
        messages = self._build_completion(user_message, user_id)

        response = await self._client.chat.completions.create(
            messages=messages, model=self._MODEL_NAME, top_p=self.TOP_P
        )

        answer = response.choices[0].message.content
        prompt, completion = response.usage.prompt_tokens, response.usage.completion_tokens
        total = prompt + completion
        logger.info(f"[User '{user_id}'] prompt: {prompt}, completion: {completion}, total: {total}")

        self._dialogue_history[user_id].update(user_message, answer, prompt, completion)
        return answer, total

    def reset(self, user_id: str):
        logger.info(f"Resetting history for user '{user_id}'")
        if user_id in self._dialogue_history:
            del self._dialogue_history[user_id]
        logger.info(f"Resetting role for user '{user_id}'")
        if user_id in self._custom_roles:
            del self._custom_roles[user_id]

    def set_role(self, user_id: str, role: str):
        self.reset(user_id)
        logger.info(f"Setting role for user '{user_id}': '{role}'")
        self._custom_roles[user_id] = role

    def get_role(self, user_id: str) -> str:
        return self._custom_roles.get(user_id, self.DEFAULT_ROLE)
