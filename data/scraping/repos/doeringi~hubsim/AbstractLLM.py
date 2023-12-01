from abc import ABC, abstractmethod
from langchain.base_language import BaseLanguageModel
from transformers import AutoTokenizer


class AbstractLLM(ABC):
    model_id: str
    tokenizer: AutoTokenizer
    llm: BaseLanguageModel
    tokenizer_arguments: dict
    model_arguments: dict

    def __init__(self):
        self.model_id = "google/flan-t5-small"
        self.tokenizer_arguments = {}
        self.model_arguments = None

    @property
    def get_model_id(self) -> str:
        return self.model_id

    @property
    def get_model_arguments(self) -> dict:
        return self.model_arguments

    @property
    def get_tokenizer_arguments(self) -> dict:
        return self.tokenizer_arguments

    @property
    def get_llm(self):
        return self.llm

    @property
    def get_tokenizer(self):
        return self.tokenizer

    @abstractmethod
    def download_llm(self):
        raise NotImplementedError('Abstract method "download_llm" must be implemented')

    @abstractmethod
    def load_llm(self) -> BaseLanguageModel:
        raise NotImplementedError('Abstract method "load_llm" must be implemented')
