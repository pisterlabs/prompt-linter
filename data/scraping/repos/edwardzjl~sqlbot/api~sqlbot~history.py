from uuid import uuid4

from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.schema import BaseMessage

from sqlbot.utils import utcnow


class CustomRedisChatMessageHistory(RedisChatMessageHistory):
    """Persist extra information in `additional_kwargs`."""

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Redis"""
        additional_info = {
            "id": uuid4().hex,
            "sent_at": utcnow().isoformat(),
            "type": "text",
        }
        message.additional_kwargs = additional_info | message.additional_kwargs
        super().add_message(message)
