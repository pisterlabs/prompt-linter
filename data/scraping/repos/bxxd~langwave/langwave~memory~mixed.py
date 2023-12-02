from typing import Any, Dict, List
from types import SimpleNamespace

from langchain.schema import BaseChatMessageHistory, BaseMemory
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage

from pydantic import BaseModel

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage

from langwave.memory import VolatileChatMemory, FewshotChatMemory


class MixedChatMemory(BaseChatMessageHistory, BaseModel):
    """Holder for multiple types of memory with a volatile memory for things that are dynamically added.
    Fewshot does not change, but volatile does."""

    """ use memories to hold other memory types, assuming they are all chat history"""

    fewshot_memory: FewshotChatMemory = FewshotChatMemory()
    _volatile_memory: VolatileChatMemory = VolatileChatMemory()

    @property
    def messages(self) -> List[BaseMessage]:
        self.fewshot_memory + self._volatile_memory.messages

    @messages.setter
    def messages(self, value: List[BaseMessage]):
        if not all(isinstance(m, BaseMessage) for m in value):
            raise ValueError("All elements must be instances of BaseMessage")
        self._volatile_memory.messages = value

    def add_user_message(self, message: str) -> None:
        self._volatile_memory.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        self._volatile_memory.add_ai_message(message)

    def add_message(self, message: BaseMessage) -> None:
        return self._volatile_memory.add_message(message)

    def clear(self) -> None:
        self._volatile_memory.clear()
