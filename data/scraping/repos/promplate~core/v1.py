from functools import cached_property
from typing import Any, Callable, ParamSpec, TypeVar

from openai import AsyncOpenAI, OpenAI  # type: ignore

from ...prompt.chat import Message, ensure
from ..base import *

P = ParamSpec("P")
T = TypeVar("T")


def with_params(_: Callable[P, Any]):
    """provide type hints with no runtime overhead"""

    def inherit(_: type[T]) -> Callable[..., Callable[P, T]]:
        return lambda as_is: as_is

    return inherit


inherit = with_params(OpenAI)


class Config(Configurable):
    @cached_property
    def client(self):
        return OpenAI(**self._config)

    @cached_property
    def aclient(self):
        return AsyncOpenAI(**self._config)


@inherit(Complete)
class TextComplete(Config):
    def __call__(self, text: str, /, **config):
        config = config | {"stream": False, "prompt": text}
        result = self.client.completions.create(**config)
        return result.choices[0].text


@inherit(AsyncComplete)
class AsyncTextComplete(Config):
    async def __call__(self, text: str, /, **config):
        config = config | {"stream": False, "prompt": text}
        result = await self.aclient.completions.create(**config)
        return result.choices[0].text


@inherit(Generate)
class TextGenerate(Config):
    def __call__(self, text: str, /, **config):
        config = config | {"stream": True, "prompt": text}
        stream = self.client.completions.create(**config)
        for event in stream:
            yield event.choices[0].text


@inherit(AsyncGenerate)
class AsyncTextGenerate(Config):
    async def __call__(self, text: str, /, **config):
        config = config | {"stream": True, "prompt": text}
        stream = await self.aclient.completions.create(**config)
        async for event in stream:
            yield event.choices[0].text


@inherit(Complete)
class ChatComplete(Config):
    def __call__(self, messages: list[Message] | str, /, **config):
        messages = ensure(messages)
        config = config | {"stream": False, "messages": messages}
        result = self.client.chat.completions.create(**config)
        return result.choices[0].message.content


@inherit(AsyncComplete)
class AsyncChatComplete(Config):
    async def __call__(self, messages: list[Message] | str, /, **config):
        messages = ensure(messages)
        config = config | {"stream": False, "messages": messages}
        result = await self.aclient.chat.completions.create(**config)
        return result.choices[0].message.content


@inherit(Generate)
class ChatGenerate(Config):
    def __call__(self, messages: list[Message] | str, /, **config):
        messages = ensure(messages)
        config = config | {"stream": True, "messages": messages}
        stream = self.client.chat.completions.create(**config)
        for event in stream:
            yield event.choices[0].delta.content or ""


@inherit(AsyncGenerate)
class AsyncChatGenerate(Config):
    async def __call__(self, messages: list[Message] | str, /, **config):
        messages = ensure(messages)
        config = config | {"stream": True, "messages": messages}
        stream = await self.aclient.chat.completions.create(**config)
        async for event in stream:
            yield event.choices[0].delta.content or ""
