from uuid import uuid4

from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.schema import BaseMessage

from pybot.context import session_id
from pybot.utils import utcnow


class ContextAwareMessageHistory(RedisChatMessageHistory):
    """Context aware history which also persists extra information in `additional_kwargs`."""

    @property
    def key(self) -> str:
        """Construct the record key to use"""
        return self.key_prefix + (session_id.get() or self.session_id)

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Redis"""
        additional_info = {
            "id": uuid4().hex,
            "sent_at": utcnow().isoformat(),
            "type": "text",
        }
        message.additional_kwargs = additional_info | message.additional_kwargs
        super().add_message(message)
