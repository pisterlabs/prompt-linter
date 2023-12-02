from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Literal, Union, cast

import json
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult

from opencopilot.domain.chat.entities import LoadingMessage


# TODO If used by two LLM runs in parallel this won't work as expected


class CustomAsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """Callback handler that returns an async iterator."""

    queue: asyncio.Queue[str]

    done: asyncio.Event

    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(self) -> None:
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()

    async def on_custom_loading_message(self, loading_message: LoadingMessage) -> None:
        self.queue.put_nowait(
            json.dumps({"loading_message": loading_message.to_dict()})
        )

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # If two calls are made in a row, this resets the state
        self.done.clear()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None and token != "":
            self.queue.put_nowait(json.dumps({"token": token}))

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.done.set()

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.done.set()

    async def aiter(self) -> AsyncIterator[str]:
        while not self.queue.empty() or not self.done.is_set():
            # Wait for the next token in the queue,
            # but stop waiting if the done event is set
            done, other = await asyncio.wait(
                [
                    # NOTE: If you add other tasks here, update the code below,
                    # which assumes each set has exactly one task each
                    asyncio.ensure_future(self.queue.get()),
                    asyncio.ensure_future(self.done.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the other task
            if other:
                other.pop().cancel()

            # Extract the value of the first completed task
            json_or_done = cast(Union[str, Literal[True]], done.pop().result())

            # If the extracted value is the boolean True, the done event was set
            if json_or_done is True:
                break

            # Otherwise, the extracted value is a token, which we yield
            yield json_or_done
