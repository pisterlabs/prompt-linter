from typing import Optional

from .base import BaseAdapter
from .client_mixins import AnthropicAdapterMixin, OpenAIAdapterMixin

__all__ = ["AnthropicChatbotAdapter", "BaseChatbotAdapter", "OpenAIChatbotAdapter"]


class BaseChatbotAdapter(BaseAdapter):
    def __init__(self, chat_history_length: int = 5, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.chat_history_length = chat_history_length
        self.chat_history = self.client.create_chat_history(max_length=self.chat_history_length)


class AnthropicChatbotAdapter(AnthropicAdapterMixin, BaseChatbotAdapter):
    """Adapter for an Anthropic chatbot."""


class OpenAIChatbotAdapter(OpenAIAdapterMixin, BaseChatbotAdapter):
    """Adapter for an OpenAI chatbot."""
