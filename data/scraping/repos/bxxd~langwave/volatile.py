from typing import Any, Dict, List

# from langchain.schema import BaseMemory
# from langchain.memory.chat_memory import BaseChatMemory
# from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory import ChatMessageHistory


class VolatileChatMemory(ChatMessageHistory):
    """An in-memory store that lives for the length of the process."""

    pass
