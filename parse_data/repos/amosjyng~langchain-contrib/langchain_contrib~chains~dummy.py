"""Module for dummy chains."""


from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate

from langchain_contrib.llms import DummyLanguageModel
from langchain_contrib.prompts import DummyPromptTemplate


class DummyLLMChain(LLMChain):
    """A dummy LLMChain for when you need an LLMChain but don't care for a real one."""

    prompt: BasePromptTemplate = DummyPromptTemplate()
    llm: BaseLanguageModel = DummyLanguageModel()
