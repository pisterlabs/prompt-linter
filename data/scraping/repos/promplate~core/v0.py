from importlib.metadata import metadata
from typing import TYPE_CHECKING, Any

import openai
from openai import ChatCompletion, Completion  # type: ignore

from ...prompt.chat import Message, ensure
from ..base import *

meta = metadata("promplate")

if openai.api_key is None:
    openai.api_key = ""

openai.app_info = (openai.app_info or {}) | {  # type: ignore
    "name": "Promplate",
    "version": meta["version"],
    "url": meta["home-page"],
}


if TYPE_CHECKING:

    class Config(Configurable):
        def __init__(
            self,
            model: str,
            temperature: float | int | None = None,
            top_p: float | int | None = None,
            stop: str | list[str] | None = None,
            max_tokens: int | None = None,
            api_key: str | None = None,
            api_base: str | None = None,
            **other_config,
        ):
            self.model = model
            self.temperature = temperature
            self.top_p = top_p
            self.stop = stop
            self.max_tokens = max_tokens
            self.api_key = api_key
            self.api_base = api_base

            for key, val in other_config.items():
                setattr(self, key, val)

        def __setattr__(self, *_):
            ...

        def __getattr__(self, _):
            ...

else:
    Config = Configurable


class TextComplete(Config, Complete):
    def __call__(self, text: str, /, **config):
        config = self._config | config | {"stream": False, "prompt": text}
        result: Any = Completion.create(**config)
        return result["choices"][0]["text"]


class AsyncTextComplete(Config, AsyncComplete):
    async def __call__(self, text: str, /, **config):
        config = self._config | config | {"stream": False, "prompt": text}
        result: Any = await Completion.acreate(**config)
        return result["choices"][0]["text"]


class TextGenerate(Config, Generate):
    def __call__(self, text: str, /, **config):
        config = self._config | config | {"stream": True, "prompt": text}
        stream: Any = Completion.create(**config)
        for event in stream:
            yield event["choices"][0]["text"]


class AsyncTextGenerate(Config, AsyncGenerate):
    async def __call__(self, text: str, /, **config):
        config = self._config | config | {"stream": True, "prompt": text}
        stream: Any = await Completion.acreate(**config)
        async for event in stream:
            yield event["choices"][0]["text"]


class ChatComplete(Config, Complete):
    def __call__(self, messages: list[Message] | str, /, **config):
        messages = ensure(messages)
        config = self._config | config | {"stream": False, "messages": messages}
        result: Any = ChatCompletion.create(**config)
        return result["choices"][0]["message"]["content"]


class AsyncChatComplete(Config, AsyncComplete):
    async def __call__(self, messages: list[Message] | str, /, **config):
        messages = ensure(messages)
        config = self._config | config | {"stream": False, "messages": messages}
        result: Any = await ChatCompletion.acreate(**config)
        return result["choices"][0]["message"]["content"]


class ChatGenerate(Config, Generate):
    def __call__(self, messages: list[Message] | str, /, **config):
        messages = ensure(messages)
        config = self._config | config | {"stream": True, "messages": messages}
        stream: Any = ChatCompletion.create(**config)
        for event in stream:
            delta: dict = event["choices"][0]["delta"]
            yield delta.get("content", "")


class AsyncChatGenerate(Config, AsyncGenerate):
    async def __call__(self, messages: list[Message] | str, /, **config):
        messages = ensure(messages)
        config = self._config | config | {"stream": True, "messages": messages}
        stream: Any = await ChatCompletion.acreate(**config)
        async for event in stream:
            delta: dict = event["choices"][0]["delta"]
            yield delta.get("content", "")
