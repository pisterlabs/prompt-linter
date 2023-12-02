from typing import Literal, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openai.types.beta import Thread
from openai.types.beta.thread_create_and_run_params import ThreadMessage
from openai.types.beta.thread_deleted import ThreadDeleted

app = APIRouter()
ai = AsyncOpenAI()


@app.get("/api/thread", response_model=Thread)
async def create_thread():
    """
    Create a new thread.
    """
    threads = ai.beta.threads
    response = await threads.create()
    return response


@app.delete("/api/thread/{thread_id}", response_model=ThreadDeleted)
async def delete_thread(*, thread_id: str):
    """
    Delete a thread.
    """
    threads = ai.beta.threads
    response = await threads.delete(thread_id=thread_id)
    return response


@app.post("/api/messages/{thread_id}", response_model=ThreadMessage)
async def create_message(
    *,
    content: str,
    thread_id: str,
    role: Literal["user"] = "user",
    file_ids: list[str] = [],
    metadata: Optional[dict[str, str]] = {},
):
    """
    Create a message.
    """
    messages = ai.beta.threads.messages
    response = await messages.create(
        thread_id=thread_id,
        content=content,
        role=role,
        file_ids=file_ids,
        metadata=metadata,
    )
    return response


@app.get("/api/messages/{thread_id}", response_class=StreamingResponse)
async def retrieve_messages(*, thread_id: str):
    """
    Retrieve messages.
    """
    messages = ai.beta.threads.messages
    response = await messages.list(thread_id=thread_id)

    async def generator():
        async for message in response:
            yield f"data: {message.json()}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
