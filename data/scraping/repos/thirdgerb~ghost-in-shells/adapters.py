from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from typing import Dict, List

import openai
from pydantic import BaseModel, Field

from ghoshell.ghost import ContextError
from ghoshell.llms.contracts import LLMTextCompletion
from ghoshell.llms.openai_contracts import OpenAIChatCompletion, OpenAIChatChoice, OpenAIChatMsg, OpenAIFuncSchema

proxy_env = os.getenv("OPENAI_PROXY", "")
if proxy_env:
    openai.proxy = {"https": proxy_env}


class TextCompletionConfig(BaseModel):
    # text completion configs
    # 慢慢完善.
    model: str = "text-davinci-003"
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: float = 30
    request_timeout: float = 5

    def text_completion_kwargs(self) -> Dict:
        return self.model_dump()


class ChatCompletionConfig(BaseModel):
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 30
    request_timeout: float = 10

    def chat_completion_kwargs(self) -> Dict:
        return self.model_dump()


class OpenAIConfig(BaseModel):
    text_completions: Dict[str, TextCompletionConfig] = Field(
        default_factory=lambda: {"default": TextCompletionConfig()}
    )
    chat_completions: Dict[str, ChatCompletionConfig] = Field(
        default_factory=lambda: {"default": ChatCompletionConfig()}
    )


class OpenAITextCompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str


class OpenAITokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAITextCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAITextCompletionChoice]
    usage: OpenAITokenUsage


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatChoice]
    usage: OpenAITokenUsage


class OpenAIRecordStorage(metaclass=ABCMeta):

    @abstractmethod
    def record(self, request: Dict, response: Dict | None, err: Exception | None) -> None:
        pass


class OpenAIAdapter(LLMTextCompletion, OpenAIChatCompletion):
    """
    openai 套皮实现
    """

    def __init__(self, config: OpenAIConfig, storage: OpenAIRecordStorage):
        self._config = config
        self._storage = storage

    @classmethod
    def contracts(cls) -> List:
        return [LLMTextCompletion, OpenAIChatCompletion]

    def text_completion(self, prompt: str, config_name: str = "") -> str:
        if not config_name:
            config_name = "default"
        completion_config = self._config.text_completions.get(config_name, None)
        if completion_config is None:
            raise RuntimeError(f"completion config {config_name} not found")
        return self._run_text_completion(prompt, completion_config)

    def _run_text_completion(self, prompt: str, config: TextCompletionConfig) -> str:
        if not prompt:
            raise RuntimeError("prompt shall not be none")
        request = config.text_completion_kwargs()
        resp = None
        err = None
        try:
            resp = openai.Completion.create(
                prompt=prompt,
                **request,
            )
        except openai.error.OpenAIError as e:
            err = ContextError(str(e))
            err.with_traceback(e.__traceback__)
            raise err
        finally:
            self._storage.record(request, resp, err)

        parsed = OpenAITextCompletionResponse(**resp.to_dict_recursive())
        return parsed.choices[0].text

    def chat_completion(
            self,
            session_id: str,
            chat_context: List[OpenAIChatMsg],
            functions: List[OpenAIFuncSchema] | None = None,
            function_call: str = "",
            config_name: str = "",  # 选择哪个预设的配置
    ) -> OpenAIChatChoice:
        config_name = config_name if config_name else "default"
        config = self._config.chat_completions.get(config_name, None)
        if config is None:
            raise RuntimeError(f"chat completion config {config_name} not found")

        request = None
        resp_dict = None
        err = None
        try:
            request = config.chat_completion_kwargs()

            messages: List[Dict] = []
            for msg in chat_context:
                messages.append(msg.to_message())
            request["messages"] = messages

            # functions
            if functions:
                request["functions"] = [func.dict() for func in functions]

            # function_call
            if functions:
                if function_call == "none":
                    request["function_call"] = "none"
                elif function_call:
                    request["function_call"] = {"name": function_call}
                else:
                    request["function_call"] = "auto"

            resp = openai.ChatCompletion.create(**request)
            resp_dict = resp.to_dict_recursive()
        except openai.error.OpenAIError as e:
            err = ContextError(str(e))
            err.with_traceback(e.__traceback__)
            raise err
        finally:
            self._storage.record(request, resp_dict, err)

        resp = OpenAIChatCompletionResponse(**resp_dict)
        return resp.choices[0]
