"""Bodhilib OpenAI plugin LLM Service module."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from bodhilib import BaseLLM, Prompt, PromptStream, Role, Service, prompt_output, service_provider

import openai
from openai.openai_response import OpenAIResponse

from ._version import __version__

"""OpenAI LLM module."""


class OpenAIChat(BaseLLM):
    """OpenAI Chat API implementation for :class:`~bodhilib.LLM`."""

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs

    def _generate(
        self,
        prompts: List[Prompt],
        *,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        user: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> Union[Prompt, PromptStream]:
        """Bodhilib LLM service implementation for OpenAI Chat API.

        Returns:
            Union[:class:`~bodhilib.Prompt`, :class:`~bodhilib.PromptStream`]: response from LLM
                as :class:`~bodhilib.Prompt` or :class:`~bodhilib.PromptStream`
        """
        all_args = {
            **self.kwargs,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": kwargs.get("logit_bias", None),
            "user": user,
            **kwargs,
        }
        all_args = {k: v for k, v in all_args.items() if v is not None}
        if "model" not in all_args:
            raise ValueError("parameter model is required")
        messages = self._to_messages(prompts)
        response = openai.ChatCompletion.create(messages=messages, **all_args)
        if "stream" in all_args and all_args["stream"]:
            return PromptStream(response, _chat_response_to_prompt_transformer)
        content = response["choices"][0]["message"]["content"]
        return prompt_output(content)

    def _to_messages(self, prompts: List[Prompt]) -> List[Dict[str, str]]:
        role_lookup = {Role.SYSTEM.value: "system", Role.AI.value: "assistant", Role.USER.value: "user"}
        return [{"role": role_lookup[p.role.value], "content": p.text} for p in prompts]


class OpenAIText(BaseLLM):
    """Bodhilib LLM service implementation for OpenAI Text API."""

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs

    def _generate(
        self,
        prompts: List[Prompt],
        *,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[Prompt, PromptStream]:
        """Bodhilib LLM service implementation for OpenAI Text API.

        Returns:
            Union[:class:`~bodhilib.Prompt`, :class:`~bodhilib.PromptStream`]: response from LLM
                as :class:`~bodhilib.Prompt` or :class:`~bodhilib.PromptStream`
        """
        prompt = self._to_prompt(prompts)
        all_args = {
            **self.kwargs,
            "stream": stream,
            "suffix": kwargs.pop("suffix", None),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "logprobs": kwargs.pop("logprobs", None),
            "echo": kwargs.pop("echo", None),
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "best_of": kwargs.pop("best_of", None),
            "logit_bias": kwargs.get("logit_bias", None),
            "user": user,
            **kwargs,
        }
        all_args = {k: v for k, v in all_args.items() if v is not None}
        if "model" not in all_args:
            raise ValueError("parameter model is required")
        response = openai.Completion.create(prompt=prompt, **all_args)
        if "stream" in all_args and all_args["stream"]:
            return PromptStream(response, _text_response_to_prompt_transfromer)
        return _text_response_to_prompt_transfromer(response)

    def _to_prompt(self, prompts: List[Prompt]) -> str:
        return "\n".join([p.text for p in prompts])


def _chat_response_to_prompt_transformer(response: OpenAIResponse) -> Prompt:
    result = response["choices"][0]
    content = "" if result["finish_reason"] else result["delta"]["content"]
    return prompt_output(content)


def _text_response_to_prompt_transfromer(response: OpenAIResponse) -> Prompt:
    result = response["choices"][0]["text"]
    return prompt_output(result)


@service_provider
def bodhilib_list_services() -> List[Service]:
    """Return a list of services supported by the plugin."""
    return [
        Service(
            service_name="openai_chat",
            service_type="llm",
            publisher="bodhiext",
            service_builder=openai_chat_service_builder,
            version=__version__,
        ),
        Service(
            service_name="openai_text",
            service_type="llm",
            publisher="bodhiext",
            service_builder=openai_text_service_builder,
            version=__version__,
        ),
    ]


def openai_text_service_builder(
    *,
    service_name: Optional[str] = None,
    service_type: Optional[str] = "llm",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> OpenAIText:
    """Returns an instance of OpenAIText LLM for the given arguments.

    Args:
        service_name (Optional[str]): service name to wrap, should be "openai_text"
        service_type (Optional[str]): service of the implementation, should be "llm"
        model (Optional[str]): OpenAI model identifier, e.g. text-ada-002
        api_key (Optional[str]): OpenAI api key, if not set, it will be read from environment variable OPENAI_API_KEY
        **kwargs: additional pass through arguments for OpenAI API client

    Returns:
        OpenAIText: an instance of OpenAIText implementing :class:`bodhilib.LLM`

    Raises:
        ValueError: if service_name is not "openai_text" or service_type is not "llm"
        ValueError: if api_key is not set and environment variable OPENAI_API_KEY is not set
    """
    if service_name != "openai_text" or service_type != "llm":
        raise ValueError(
            f"Unknown params: {service_name=}, {service_type=}, supported params: service_name='openai_text',"
            " service_type='llm'"
        )
    _set_openai_api_key(api_key)
    all_args: Dict[str, Any] = {"model": model, "api_key": api_key, **kwargs}
    all_args = {k: v for k, v in all_args.items() if v is not None}
    return OpenAIText(**all_args)


def openai_chat_service_builder(
    *,
    service_name: Optional[str] = None,
    service_type: Optional[str] = "llm",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> OpenAIChat:
    """Returns an instance of OpenAIChat LLM for the given arguments.

    Args:
        service_name: service name to wrap, should be "openai_chat"
        service_type: service type of the component, should be "llm"
        model: OpenAI chat model identifier, e.g. gpt-3.5-turbo
        api_key: OpenAI api key, if not set, it will be read from environment variable OPENAI_API_KEY
        **kwargs: additional arguments passed to the OpenAI API client

    Returns:
        OpenAIChat: an instance of OpenAIChat implementing :class:`bodhilib.LLM`

    Raises:
        ValueError: if service_name is not "openai" or service_type is not "llm"
        ValueError: if api_key is not set and environment variable OPENAI_API_KEY is not set
    """
    if service_name != "openai_chat" or service_type != "llm":
        raise ValueError(
            f"Unknown params: {service_name=}, {service_type=}, supported params: service_name='openai_chat',"
            " service_type='llm'"
        )
    _set_openai_api_key(api_key)
    all_args: Dict[str, Any] = {"model": model, "api_key": api_key, **kwargs}
    all_args = {k: v for k, v in all_args.items() if v is not None}
    return OpenAIChat(**all_args)


def _set_openai_api_key(api_key: Optional[str]) -> None:
    if api_key is None:
        if os.environ.get("OPENAI_API_KEY") is None:
            raise ValueError("environment variable OPENAI_API_KEY is not set")
        else:
            openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        openai.api_key = api_key
