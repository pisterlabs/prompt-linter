from typing import Any, Dict

from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM

from zamm.prompts.dummy import DummyPromptTemplate


class DummyLLMChain(LLMChain):
    def __init__(self, llm: BaseLLM):
        super().__init__(llm=llm, prompt=DummyPromptTemplate())

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        raise NotImplementedError()
