from typing import List, Optional

from ..prompters.base import Prompter
from ..prompters.classification import ClassificationPrompter, SentimentAnalysisPrompter
from .base import BaseTaskAdapter
from .client_mixins import AnthropicAdapterMixin, CohereAdapterMixin, OpenAIAdapterMixin

__all__ = [
    "AnthropicSentimentAnalysisAdapter",
    "AnthropicTextClassificationAdapter",
    "CohereSentimentAnalysisAdapter",
    "CohereTextClassificationAdapter",
    "OpenAISentimentAnalysisAdapter",
    "OpenAITextClassificationAdapter",
    "SentimentAnalysisAdapter",
    "TextClassificationAdapter",
]


class TextClassificationAdapter(BaseTaskAdapter):
    def __init__(
        self,
        categories: List[str],
        prompter: Optional[Prompter] = None,
        lowercase: bool = True,
        api_key: Optional[str] = None,
        **kwargs
    ):
        self.lowercase = lowercase
        self.categories = [c.lower() for c in categories] if lowercase else categories
        prompter = prompter or ClassificationPrompter(categories=self.categories)
        super().__init__(prompter=prompter, api_key=api_key, **kwargs)


class SentimentAnalysisAdapter(TextClassificationAdapter):
    def __init__(self, include_neutral: bool = True, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            categories=["positive", "negative"] + (["neutral"] if include_neutral else []),
            prompter=SentimentAnalysisPrompter(include_neutral=include_neutral),
            api_key=api_key,
            lowercase=True,
            **kwargs,
        )


class AnthropicTextClassificationAdapter(AnthropicAdapterMixin, TextClassificationAdapter):
    """Adapter for an Anthropic text classification model."""

    def prepare_input_content(self, content) -> dict:
        return {"prompt": content, "stop_sequences": ["\n\n"]}


class AnthropicSentimentAnalysisAdapter(AnthropicAdapterMixin, SentimentAnalysisAdapter):
    """Adapter for an Anthropic sentiment analysis model."""

    def prepare_input_content(self, content) -> dict:
        return {"prompt": content, "stop_sequences": ["\n\n"]}


class CohereTextClassificationAdapter(CohereAdapterMixin, TextClassificationAdapter):
    """Adapter for a Cohere text classification model."""


class CohereSentimentAnalysisAdapter(CohereAdapterMixin, SentimentAnalysisAdapter):
    """Adapter for a Cohere sentiment analysis model."""


class OpenAITextClassificationAdapter(OpenAIAdapterMixin, TextClassificationAdapter):
    """Adapter for an OpenAI text classification model."""


class OpenAISentimentAnalysisAdapter(OpenAIAdapterMixin, SentimentAnalysisAdapter):
    """Adapter for an OpenAI sentiment analysis model."""
