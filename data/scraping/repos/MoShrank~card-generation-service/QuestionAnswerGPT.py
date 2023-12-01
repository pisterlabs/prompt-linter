from abc import ABC, abstractmethod

import openai

from external.gpt import get_chatgpt_completion
from models.ModelConfig import Message, QuestionAnswerGPTConfig


class QuestionAnswerGPTInterface(ABC):
    @abstractmethod
    def __call__(self, documents: list[str], question: str, user_id: str) -> str:
        pass


class QuestionAnswerGPT(QuestionAnswerGPTInterface):
    def __init__(self, model_config: QuestionAnswerGPTConfig, openai_api_key: str):
        openai.api_key = openai_api_key
        self._model_config = model_config

    def __call__(self, documents: list[str], question: str, user_id: str) -> str:
        text = "\n\n".join(documents)

        system_prompt = self._model_config.system_message.format(question=question)

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=text),
        ]

        completion = get_chatgpt_completion(
            self._model_config.parameters, messages, user_id
        )

        return completion
