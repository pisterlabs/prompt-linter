"""This is an example of how to use async langchain with fastapi and return a streaming response.
The latest version of Langchain has improved its compatibility with asynchronous FastAPI,
making it easier to implement streaming functionality in your applications.
"""
import asyncio
import os
from typing import AsyncIterable, Awaitable

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel

# Two ways to load env variables
# 1.load env variables from .env file
load_dotenv()

# 2.manually set env variables
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()


async def send_message(message: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
    )

    # model = AzureChatOpenAI(
    #     openai_api_base=os.environ["AZURE_BASE_URL"],
    #     openai_api_version=os.environ["AZURE_API_VERSION"],
    #     deployment_name=os.environ["AZURE_DEPLOYMENT_NAME"],
    #     openai_api_key=os.environ["AZURE_API_KEY"],
    #     openai_api_type="azure",
    #     streaming=True,
    #     verbose=True,
    #     callbacks=[callback],
    # )

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with an event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()

    # Begin a task that runs in the background.
    task = asyncio.create_task(wrap_done(
        model.agenerate(messages=[[HumanMessage(content=message)]]),
        callback.done),
    )

    async for token in callback.aiter():
        # Use server-sent-events to stream the response
        yield f"data: {token}\n\n"

    await task


class StreamRequest(BaseModel):
    """Request body for streaming."""
    message: str


@app.post("/stream")
def stream(body: StreamRequest):
    return StreamingResponse(send_message(body.message), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8000, app=app)
