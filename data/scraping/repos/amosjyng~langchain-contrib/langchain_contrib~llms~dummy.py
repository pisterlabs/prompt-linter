"""A module for defining a dummy LLM."""

from typing import Any, Coroutine, List, Optional, Sequence

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.schema import BaseMessage, LLMResult, PromptValue


class DummyLanguageModel(BaseLanguageModel):
    """A dummy LLM for when you need an LLM but don't care for a real one.

    You can use this instead of FakeLLM when you want to be sure the LLM is not
    actually getting called.
    """

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Error out because this is a dummy LLM."""
        raise NotImplementedError("You're using the dummy LLM")

    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Error out asynchronously because this is a dummy LLM."""
        raise NotImplementedError("You're using the dummy LLM")

    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None) -> str:
        """Error out because this is a dummy LLM."""
        raise NotImplementedError("You're using the dummy LLM")

    def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None
    ) -> Coroutine[Any, Any, str]:
        """Error out because this is a dummy LLM."""
        raise NotImplementedError("You're using the dummy LLM")

    def predict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[Sequence[str]] = None
    ) -> BaseMessage:
        """Error out because this is a dummy LLM."""
        raise NotImplementedError("You're using the dummy LLM")

    def apredict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[Sequence[str]] = None
    ) -> Coroutine[Any, Any, BaseMessage]:
        """Error out because this is a dummy LLM."""
        raise NotImplementedError("You're using the dummy LLM")
