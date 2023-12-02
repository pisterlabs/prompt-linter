"""Callback handlers used in the app."""
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import BaseMessage

from models.chat_response import ChatResponse


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                                  parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                                  metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        pass

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream")
        await self.websocket.send_json(resp.model_dump())

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized["name"]
        resp = ChatResponse(sender="bot", message=f"Running tool `{tool_name}`, arguments `{input_str}`", type="tool")
        await self.websocket.send_json(resp.model_dump())

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        resp = ChatResponse(sender="bot", message=f"Tool finished, output `{output}`", type="tool")
        await self.websocket.send_json(resp.model_dump())
