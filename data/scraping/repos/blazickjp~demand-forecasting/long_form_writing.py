import os

from typing import Dict, List, Any

from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI

MODEL = 'gpt-3.5-turbo'


class LongFormWritingChain(Chain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        prompt = PromptTemplate(
            template="",
            input_variables=["conversation_history"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)