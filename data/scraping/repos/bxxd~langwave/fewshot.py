from typing import Any, Dict, List

from langchain.memory import ChatMessageHistory


class FewshotChatMemory(ChatMessageHistory):
    """An in-memory store that lives for the length of the process."""

    pass
