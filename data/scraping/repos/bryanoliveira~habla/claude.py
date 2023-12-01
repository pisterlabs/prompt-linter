import os
from typing import Iterator

import anthropic
from habla.models import BaseModel


class AnthropicModel(BaseModel):
    def __init__(
        self,
        system_message,
        model="claude-instant-v1-100k",
        max_tokens=2048,
        stream=False,
    ):
        self.system_message = system_message
        self.model = model
        self.max_tokens = max_tokens
        self.stream = stream
        self.reset_conversation()

        self.client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])

    def reset_conversation(self):
        """Resets the conversation to just the system message."""
        self.conversation = [
            f"{anthropic.HUMAN_PROMPT} {self.system_message}",
        ]

    def add_message(self, message: str, role: str):
        """Adds a message from the human or AI to the conversation."""
        assert role in [
            "user",
            "assistant",
        ], "role must be either 'user' or 'assistant'"

        if len(self.conversation) == 1:
            # only system message
            self.conversation[-1] += message
        else:
            # subsequent messages
            self.conversation.append(
                {
                    "user": anthropic.HUMAN_PROMPT + " ",
                    "assistant": anthropic.AI_PROMPT + " ",
                }[role]
                + message
            )

    def get_conversation(self) -> str:
        return "".join(self.conversation)

    def count_tokens(self, text: str) -> int:
        return anthropic.count_tokens(text)

    def respond(self) -> Iterator:
        assert len(self.conversation) > 0, "conversation must not be empty"
        assert self.conversation[-1].startswith(
            anthropic.HUMAN_PROMPT
        ), "last message must be from a human"

        prompt = self.get_conversation() + anthropic.AI_PROMPT
        response = self.client.completion_stream(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            max_tokens_to_sample=self.max_tokens,
            model=self.model,
            stream=self.stream,
        )
        partial_response = ""
        for event in response:
            if "completion" in event:
                yield event["completion"][len(partial_response) :]
                if self.stream:
                    partial_response = event["completion"]
            else:
                raise RuntimeError(f"Unknown event {event}")
