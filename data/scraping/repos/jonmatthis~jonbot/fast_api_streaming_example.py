import asyncio
import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import LLMResult

app = FastAPI()


class MyCustomSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")


class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from chatbot."""

    async def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_new_token(self, token: str, **kwargs) -> str:
        return f"async handler: token: {token}"

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is ending")


load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=25,
    streaming=True,
    callbacks=[MyCustomAsyncHandler()],
)


async def stream_generator(query: str):
    async for token in llm.astream(query):
        yield f"wowooo - {token.content}\n"


@app.get("/stream_chat/")
def stream_chat(query: str):
    return StreamingResponse(stream_generator(query), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
