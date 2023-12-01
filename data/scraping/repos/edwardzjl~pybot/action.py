from typing import Any, Optional
from uuid import UUID

from fastapi import WebSocket
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import AgentAction, BaseChatMessageHistory, SystemMessage

from pybot.schemas import ChatMessage


class AgentActionCallbackHandler(AsyncCallbackHandler):
    """Callback handler for sending tool execution information back to user."""

    def __init__(
        self,
        websocket: WebSocket,
        conversation_id: str,
        history: BaseChatMessageHistory,
    ):
        self.websocket = websocket
        self.conversation_id = conversation_id
        self.history = history

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Persist 'thought' (action log) to history.
        Action log was sent to user by streaming callback, I don't need to send it again here.
        """
        self.history.add_ai_message(action.log)

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Send 'observation' (action result) to user, and persist to history."""
        message = ChatMessage(
            id=run_id,
            conversation=self.conversation_id,
            from_="system",
            content=output,
            type="text",
        )
        await self.websocket.send_text(message.model_dump_json())
        self.history.add_message(
            SystemMessage(
                content=output,
                additional_kwargs={
                    "prefix": "<|im_start|>system-observation\n",
                    "suffix": "<|im_end|>",
                },
            )
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        message = ChatMessage(
            id=run_id,
            conversation=self.conversation_id,
            from_="system",
            content=str(error),
            type="text",
        )
        await self.websocket.send_text(message.model_dump_json())
        self.history.add_message(SystemMessage(content=str(error)))
