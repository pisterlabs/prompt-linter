from typing import Any, Dict, List, cast

from .base import BaseLlmModel, LLmInputInterface

from langchain.llms.ai21 import AI21
from langchain.schema.output import LLMResult
from langchain.callbacks.base import Callbacks, BaseCallbackManager


class AI21Model(BaseLlmModel):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = AI21(
            ai21_api_key=input.api_key,
            model=input.model_name if input.model_name else "j2-jumbo-instruct",
            temperature=input.temperature,
            maxTokens=input.max_tokens,
            topP=input.top_p,
            stop=input.stop,
            cache=input.cache,
            verbose=input.verbose,
            callback_manager=cast(BaseCallbackManager, input.callback_manager),
            metadata=input.metadata,
        )  # type: ignore

    def compelete(self, prompts: List[str], callbacks: Callbacks | List[Callbacks] = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(prompts=prompts, callbacks=callbacks, metadata=metadata)
        return result

    async def acompelete(self, prompts: List[str], callbacks: Callbacks | List[Callbacks] = None, metadata: Dict[str, Any] | None = None):
        result = await self.client.agenerate(prompts=prompts, metadata=metadata, callbacks=callbacks)
        return result