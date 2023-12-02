from __future__ import annotations

import asyncio
from typing import Any, Union, Dict
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.output import LLMResult

# TODO If used by two LLM runs in parallel this won't work as expected


class EnqueueCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queues: Dict[str,asyncio.Queue], message_id:str):
        self.queues = queues
        self.message_id = message_id
      
        
   
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.queues[self.message_id].put(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        await self.queues[self.message_id].put(None)

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        await self.queues[self.message_id].put(None)


    # TODO implement the other methods

   