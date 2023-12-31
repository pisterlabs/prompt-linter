"""
This type stub file was generated by pyright.
"""

from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel

class TaskCreationChain(LLMChain):
    """Chain generating tasks."""
    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = ...) -> LLMChain:
        """Get the response parser."""
        ...
    


