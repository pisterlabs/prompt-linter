"""Dependencies for items endpoints."""
from contextlib import suppress
import os
from typing import Any

from fastapi import WebSocket
from langchain.callbacks.base import AsyncCallbackHandler, Callbacks
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from starlette.websockets import WebSocketState


# Models
class MessageResponse(BaseModel):
    """Response model for messages."""

    message: str


class StatusResponse(BaseModel):
    """Response model for status."""

    status: str


class ErrorResponse(StatusResponse):
    """Response model for errors."""

    error_message: str


class DoneResponse(StatusResponse):
    """Response model for done."""

    result: str


class WebSocketStreamingCallback(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **_: Any) -> None:
        """Run when LLM generates a new token."""
        with suppress(
            Exception
        ):  # Suppresses `Error in WebSocketStreamingCallback.on_llm_new_token callback: received 1000 (OK); then sent 1000 (OK)`
            if self.websocket.client_state == WebSocketState.CONNECTED:
                if token != "":
                    response = MessageResponse(message=token)
                    await self.websocket.send_json(response.model_dump())


# Helper fns
def create_llm(callbacks: Callbacks, model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    """Create an LLM instance."""
    return ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=bool(callbacks),
        callbacks=callbacks,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_prompt(system: str) -> PromptTemplate:
    """Create a prompt template."""
    return PromptTemplate.from_template(template=system)


async def send_answer(
    websocket: WebSocket,
    prompt: str,
    system: str,
    callbacks: Callbacks = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.9,
    max_tokens: int = 100,
) -> DoneResponse:
    try:
        llm = create_llm(
            callbacks=callbacks,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        template = create_prompt(system=system)
        chain = LLMChain(llm=llm, prompt=template)
        result = await chain.arun(prompt)
        response = DoneResponse(result=result, status="DONE")
        await websocket.send_json(response.model_dump())
    except Exception as e:
        raise e


async def send_error(websocket: WebSocket, exception: Exception) -> None:
    error = ErrorResponse(error_message=repr(exception), status="ERROR")
    await websocket.send_json(error.model_dump())
