import asyncio
from typing import AsyncIterable, Awaitable

import openai
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.docstore.document import Document
from qdrant_client import QdrantClient

from ..config import settings
from ..utils import makechain


async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    await fn
    event.set()


async def send_message(message: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    client = QdrantClient(settings.qdrant_host, port=settings.qdrant_port)

    embeds = openai.Embedding.create(input=message, engine=settings.openai_embedding_model)

    chain = makechain(callback)

    # query the database
    resp = client.search(collection_name=settings.qdrant_collection, query_vector=embeds["data"][0]["embedding"], limit=6)

    documents = [Document(page_content=resp[i].payload["text"]) for i in range(len(resp))]
    list_urls = list(set(resp[i].payload["url"] for i in range(len(resp))))

    task = asyncio.create_task(
        wrap_done(chain.arun(input_documents=documents, question=message, links=str(list_urls)), callback.done),
    )

    # Logic to send the response to the frontend as an event stream
    async for token in callback.aiter():
        yield f"{token}"

    await task
