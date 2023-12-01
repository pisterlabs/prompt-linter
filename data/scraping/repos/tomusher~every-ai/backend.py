from dataclasses import dataclass
from typing import List, Literal, Optional

from every_ai import AIBackend, registry
from every_ai.exceptions import InvalidBackendConfigurationError

# Types
ChatModels = Literal[
    "claude-instant-1",
    "claude-2",
]


@dataclass
class ChatModelConfig:
    token_limit: int


@dataclass
class BackendConfig:
    api_key: str = ""
    chat_model: ChatModels = "claude-instant-1"


# Constants

CHAT_MODELS = {
    "claude-instant-1": ChatModelConfig(token_limit=100000),
    "claude-2": ChatModelConfig(token_limit=100000),
}

# Implementation


@registry.register("anthropic")
class AnthropicBackend(AIBackend):
    config_class = BackendConfig

    def __init__(self, config: dict):
        try:
            from anthropic import Anthropic

            self.config = self.config_class(**config)
            api_key = self.config.api_key
            self.client = Anthropic(api_key=api_key)
        except AttributeError as e:
            raise InvalidBackendConfigurationError(
                "The api_key setting must be configured to use OpenAI"
            ) from e

    def chat(
        self, *, system_messages: Optional[List[str]] = None, user_messages: List[str]
    ) -> str:
        from anthropic import AI_PROMPT, HUMAN_PROMPT

        completion = self.client.chat.completions.create(
            model=self.config.chat_model,
            prompt=f"{HUMAN_PROMPT} {' '.join(system_messages or [])} {' '.join(user_messages)}{AI_PROMPT}",
        )
        return completion.completion
