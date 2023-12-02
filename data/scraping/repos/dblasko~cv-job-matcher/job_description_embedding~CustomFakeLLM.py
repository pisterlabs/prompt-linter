
from typing import Any, List
from langchain.callbacks.manager import Callbacks
from langchain.llms import FakeListLLM
from langchain.schema import LLMResult, BaseMessage

class CustomFakeLLM(FakeListLLM):

    def __init__(self, responses: list=[]) -> None:
        super().__init__(responses=responses)

    def generate(self, messages: List[List[BaseMessage]], stop: List[str] | None = None, callbacks: Callbacks = None, *, tags: List[str] | None = None, **kwargs: Any) -> LLMResult:
        prompts = [str(i) for m in messages for i in m]
        return super().generate(prompts, stop, callbacks, tags=tags, **kwargs)