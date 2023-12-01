from typing import Any, Dict, List
from langchain.callbacks.base import AsyncCallbackHandler
from schemas import ChatResponse


class WebSocketCallbackHandler(AsyncCallbackHandler):
    """Base callback handler for websocket responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def send_response(
        self, sender: str, message: str, response_type: str
    ) -> None:
        resp = ChatResponse(sender=sender, message=message, type=response_type)
        await self.websocket.send_json(resp.dict())


class StreamingLLMCallbackHandler(WebSocketCallbackHandler):
    """Callback handler for streaming LLM responses."""

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.send_response(sender="bot", message=token, response_type="stream")


class QuestionGenCallbackHandler(WebSocketCallbackHandler):
    """Callback handler for question generation."""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        await self.send_response(
            sender="bot", message="Synthesizing question...", response_type="info"
        )
