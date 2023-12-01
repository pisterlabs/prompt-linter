
from langchain.schema import BaseChatMessageHistory
from pydantic import BaseModel

from app.web.api import get_messages_by_conversation_id, add_message_to_conversation


class SqlMessageHistory(BaseChatMessageHistory, BaseModel):
    conversation_id: str

    @property
    def messages(self):
        return get_messages_by_conversation_id(self.conversation_id)

    def add_message(self, message):
        return add_message_to_conversation(conversation_id=self.conversation_id, role=message.type,
                                           content=message.content)

    def clear(self):
        pass



