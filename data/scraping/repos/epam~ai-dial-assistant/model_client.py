from abc import ABC
from collections import defaultdict
from typing import Any, AsyncIterator, List

import openai
from aidial_sdk.chat_completion import Role
from aiohttp import ClientSession
from pydantic import BaseModel


class ReasonLengthException(Exception):
    pass


class Message(BaseModel):
    role: Role
    content: str

    def to_openai_message(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def system(cls, content):
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content):
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content):
        return cls(role=Role.ASSISTANT, content=content)


class UsagePublisher:
    def __init__(self):
        self.total_usage = defaultdict(int)

    def publish(self, usage: dict[str, int]):
        for k, v in usage.items():
            self.total_usage[k] += v

    @property
    def prompt_tokens(self) -> int:
        return self.total_usage["prompt_tokens"]

    @property
    def completion_tokens(self) -> int:
        return self.total_usage["completion_tokens"]


class ModelClient(ABC):
    def __init__(
        self,
        model_args: dict[str, Any],
        buffer_size: int,
    ):
        self.model_args = model_args
        self.buffer_size = buffer_size

    async def agenerate(
        self, messages: List[Message], usage_publisher: UsagePublisher
    ) -> AsyncIterator[str]:
        async with ClientSession(read_bufsize=self.buffer_size) as session:
            openai.aiosession.set(session)

            model_result = await openai.ChatCompletion.acreate(
                **self.model_args,
                messages=[message.to_openai_message() for message in messages]
            )

            async for chunk in model_result:  # type: ignore
                usage = chunk.get("usage")
                if usage:
                    usage_publisher.publish(usage)

                choice = chunk["choices"][0]
                text = choice["delta"].get("content")
                if text:
                    yield text

                if choice.get("finish_reason") == "length":
                    raise ReasonLengthException()
