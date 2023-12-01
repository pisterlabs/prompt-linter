from typing import Any, Dict, List

from .base import BaseChat, BaseLlmModel, LLmInputInterface

from langchain.llms.cohere import Cohere
from langchain.chat_models.cohere import ChatCohere
from langchain.schema import BaseMessage
from langchain.schema.output import LLMResult
from langchain.callbacks.base import Callbacks


class CohereModel(BaseLlmModel):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = Cohere(
            cohere_api_key=input.api_key,
            model=input.model_name if input.model_name else "command",
            k=input.top_k,
            temperature=input.temperature,
            max_tokens=input.max_tokens,
            frequency_penalty=input.repeat_penalty,
            max_retries=input.max_retries,
            stop=input.stop,
            cache=input.cache,
            verbose=input.verbose,
            callbacks=input.callbacks,
            metadata=input.metadata,
        )  # type: ignore

    def compelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(
            prompts=prompts, callbacks=callbacks, metadata=metadata)
        return result

    async def acompelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None):
        result = await self.client.agenerate(prompts=prompts, metadata=metadata, callbacks=callbacks)
        return result


class ChatCohereModel(BaseChat):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = ChatCohere(
            cohere_api_key=input.api_key,
            model=input.model_name if input.model_name else "command",
            temperature=input.temperature,
            stop=input.stop,
            cache=input.cache,
            verbose=input.verbose,
            callbacks=input.callbacks,
            metadata=input.metadata,
        )  # type: ignore

    def compelete(self, prompts: List[List[BaseMessage]], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(
            messages=prompts, callbacks=callbacks, metadata=metadata)
        return result

    async def acompelete(self, prompts: List[List[BaseMessage]], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = await self.client.agenerate(messages=prompts, callbacks=callbacks, metadata=metadata)
        return result
