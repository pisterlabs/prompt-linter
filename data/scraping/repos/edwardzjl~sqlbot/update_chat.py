from typing import Any, Optional
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.agent import AgentFinish

from sqlbot.models import Conversation
from sqlbot.utils import utcnow


class UpdateConversationCallbackHandler(AsyncCallbackHandler):
    def __init__(self, conversation_id: str):
        self.conversation_id: str = conversation_id

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent end."""
        conv = await Conversation.get(self.conversation_id)
        conv.updated_at = utcnow()
        await conv.save()
