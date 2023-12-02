from asyncio import Queue
from typing import Any, Dict, List

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult


class LLMCallback(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.rq = Queue()
        self.generate_data = ""

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.rq = Queue(1000)
        self.generate_data = ""

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print("put token: ", token)
        await self.rq.put(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        print("end token: ", response)
        await self.rq.put("<END>")

    async def get_data(self) -> str:
        _temp = await self.rq.get()
        self.generate_data += _temp
        print("get data: ", _temp)
        return "<END>" if _temp == "<END>" else self.generate_data
