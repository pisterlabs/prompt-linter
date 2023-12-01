"""LLM implementation for Cohere."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from bodhilib import LLM, BaseLLM, Prompt, PromptStream, Service, prompt_output, service_provider

import cohere
from cohere.responses.generation import StreamingText

from ._version import __version__


class Cohere(BaseLLM):
    """Cohere API implementation for :class:`~bodhilib.LLM`."""

    def __init__(
        self,
        client: Optional[cohere.Client] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ):
        """Initialize Cohere LLM service.
        
        Args:
            client (Optional[:class:`~cohere.Client`]): Pass Cohere client instance directly to be used
            model: Cohere model identifier
            api_key: api key for Cohere service, if not set, it will be read from environment variable COHERE_API_KEY
            **kwargs: additional arguments to be passed to Cohere client
        """
        all_args = {"model": model, "api_key": api_key, **kwargs}
        self.kwargs = {k: v for k, v in all_args.items() if v is not None}

        if client:
            self.client = client
        else:
            allowed_args = [
                "api_key",
                "num_workers",
                "request_dict",
                "check_api_key",
                "client_name",
                "max_retries",
                "timeout",
                "api_url",
            ]
            args = {k: v for k, v in self.kwargs.items() if k in allowed_args}
            self.client = cohere.Client(**args)

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
        if len(prompts) == 0:
            raise ValueError("Prompt is empty")
        input = self._to_cohere_prompt(prompts)
        if input == "":
            raise ValueError("Prompt is empty")
        all_args = {
            **self.kwargs,
            "stream": stream,
            "num_generations": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "p": top_p,
            "k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop_sequences": stop,
            "user": user,
            **kwargs,
        }
        all_args = {k: v for k, v in all_args.items() if v is not None}
        if "model" not in all_args:
            raise ValueError("parameter model is required")

        allowed_args = [
            "prompt",
            "prompt_vars",
            "model",
            "preset",
            "num_generations",
            "max_tokens",
            "temperature",
            "k",
            "p",
            "frequency_penalty",
            "presence_penalty",
            "end_sequences",
            "stop_sequences",
            "return_likelihoods",
            "truncate",
            "logit_bias",
            "stream",
            "user",
        ]
        args = {k: v for k, v in all_args.items() if k in allowed_args}

        response = self.client.generate(input, **args)

        if "stream" in all_args and all_args["stream"]:
            return PromptStream(response, _cohere_stream_to_prompt_transformer)
        text = response.generations[0].text
        return prompt_output(text)

    def _to_cohere_prompt(self, prompts: List[Prompt]) -> str:
        return "\n".join([p.text for p in prompts])


def _cohere_stream_to_prompt_transformer(chunk: StreamingText) -> Prompt:
    return prompt_output(chunk.text)


@service_provider
def bodhilib_list_services() -> List[Service]:
    """This function is used by bodhilib to find all services in this module."""
    return [
        Service(
            service_name="cohere",
            service_type="llm",
            publisher="bodhiext",
            service_builder=cohere_llm_service_builder,
            version=__version__,
        )
    ]


def cohere_llm_service_builder(
    *,
    service_name: Optional[str] = None,
    service_type: Optional[str] = "llm",
    client: Optional[cohere.Client] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> LLM:
    """Returns an instance of Cohere LLM service implementing :class:`~bodhilib.LLM`.

    Args:
        service_name: service name to wrap, should be "cohere"
        service_type: service type of the implementation, should be "llm"
        client (Optional[:class:`~cohere.Client`]): Pass Cohere client instance directly to be used
        model: Cohere model identifier
        api_key: api key for Cohere service, if not set, it will be read from environment variable COHERE_API_KEY
    Returns:
        :class:`~bodhilib.LLM`: a service instance implementing :class:`~bodhilib.LLM` for the given service and model
    Raises:
        ValueError: if service_name is not "cohere"
        ValueError: if service_type is not "llm"
        ValueError: if model is not set
        ValueError: if api_key is not set, and environment variable COHERE_API_KEY is not set
    """
    # TODO use pydantic for parameter validation
    if service_name != "cohere" or service_type != "llm":
        raise ValueError(
            f"Unknown params: {service_name=}, {service_type=}, supported params: service_name='cohere',"
            " service_type='llm'"
        )
    if api_key is None:
        if os.environ.get("COHERE_API_KEY") is None:
            raise ValueError("environment variable COHERE_API_KEY is not set")
        else:
            api_key = os.environ["COHERE_API_KEY"]
    return Cohere(client=client, model=model, api_key=api_key, **kwargs)
