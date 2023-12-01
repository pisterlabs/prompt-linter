from enum import Enum
from typing import Callable, List, Set, Tuple

import anthropic

from aidial_adapter_bedrock.llm.chat_emulation.history import (
    FormattedMessage,
    History,
    is_important_message,
)
from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from aidial_adapter_bedrock.utils.list import exclude_indices


class RolePrompt(str, Enum):
    HUMAN = anthropic.HUMAN_PROMPT
    AI = anthropic.AI_PROMPT


STOP_SEQUENCES: List[str] = [anthropic.HUMAN_PROMPT]


def _format_message(message: BaseMessage) -> str:
    role = (
        RolePrompt.HUMAN
        if isinstance(message, (SystemMessage, HumanMessage))
        else RolePrompt.AI
    )
    return (role + " " + message.content.lstrip()).rstrip()


class ClaudeChatHistory(History):
    def trim(
        self,
        count_tokens: Callable[[str], int],
        max_prompt_tokens: int,
    ) -> Tuple["ClaudeChatHistory", int]:
        message_tokens = [
            count_tokens(message.text) for message in self.messages
        ]
        prompt_tokens = sum(message_tokens)
        if prompt_tokens <= max_prompt_tokens:
            return self, 0

        discarded_messages: Set[int] = set()
        for index, message in enumerate(self.messages):
            if message.is_important:
                continue

            discarded_messages.add(index)
            prompt_tokens -= message_tokens[index]
            if prompt_tokens <= max_prompt_tokens:
                return ClaudeChatHistory(
                    messages=exclude_indices(self.messages, discarded_messages)
                ), len(discarded_messages)

        if discarded_messages:
            raise ValidationError(
                f"The token size of system messages and the last user message ({prompt_tokens}) exceeds"
                f" prompt token limit ({max_prompt_tokens})."
            )

        raise ValidationError(
            f"Prompt token size ({prompt_tokens}) exceeds prompt token limit ({max_prompt_tokens})."
        )

    @classmethod
    def create(cls, messages: List[BaseMessage]) -> "ClaudeChatHistory":
        formatted_messages = []

        for index, message in enumerate(messages):
            formatted_messages.append(
                FormattedMessage(
                    text=_format_message(message),
                    source_message=message,
                    is_important=is_important_message(messages, index),
                )
            )

        formatted_messages.append(
            FormattedMessage(text=_format_message(AIMessage(content="")))
        )

        return cls(messages=formatted_messages)
