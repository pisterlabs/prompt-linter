# -*- coding: utf-8 -*-
from enum import Enum
from logging import getLogger
from typing import Any, Optional, Sequence, Callable, Coroutine
import openai
import openai.error

from lbgpt.allocation import (
    random_allocation_function,
    max_headroom_allocation_function,
)
from lbgpt.base import _BaseGPT

logger = getLogger(__name__)


class ChatGPT(_BaseGPT):
    def __init__(
        self,
        api_key: str,
        max_parallel_calls: int = 5,
        cache: Optional[Any] = None,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
        max_cache_size: Optional[int] = 1_000,
        limit_tpm: Optional[int] = None,
        limit_rpm: Optional[int] = None,
    ):
        super().__init__(
            cache=cache,
            max_parallel_calls=max_parallel_calls,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            max_cache_size=max_cache_size,
            limit_tpm=limit_tpm,
            limit_rpm=limit_rpm,
        )
        self.api_key = api_key

    async def chat_completion(self, **kwargs) -> openai.ChatCompletion:
        # one request to the OpenAI API respecting their ratelimit

        async with self.semaphore:
            return await openai.ChatCompletion.acreate(
                api_key=self.api_key,
                **kwargs,
            )


class AzureGPT(_BaseGPT):
    def __init__(
        self,
        api_key: str,
        azure_api_base: str,
        azure_model_map: dict[str, str],
        cache: Optional[Any] = None,
        azure_openai_version: str = "2023-05-15",
        azure_openai_type: str = "azure",
        max_parallel_calls: int = 5,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
        max_cache_size: Optional[int] = 1_000,
        limit_tpm: Optional[int] = None,
        limit_rpm: Optional[int] = None,
    ):
        super().__init__(
            cache=cache,
            max_parallel_calls=max_parallel_calls,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            max_cache_size=max_cache_size,
            limit_tpm=limit_tpm,
            limit_rpm=limit_rpm,
        )

        self.api_key = api_key
        self.azure_api_base = azure_api_base
        self.azure_model_map = azure_model_map
        self.azure_openai_version = azure_openai_version
        self.azure_openai_type = azure_openai_type

    async def chat_completion(self, **kwargs) -> openai.ChatCompletion:
        """One request to the Azure OpenAI API respecting their ratelimit
        # needs to change the model parameter to deployment id
        """

        model = kwargs.pop("model")
        deployment_id = self.azure_model_map[model]
        kwargs["deployment_id"] = deployment_id

        async with self.semaphore:
            return await openai.ChatCompletion.acreate(
                api_key=self.api_key,
                api_base=self.azure_api_base,
                api_type=self.azure_openai_type,
                api_version=self.azure_openai_version,
                **kwargs,
            )


ALLOCATION_FUNCTIONS = {
    "random": random_allocation_function,
    "max_headroom": max_headroom_allocation_function,
}


class MultiLoadBalancedGPT(_BaseGPT):
    def __init__(
        self,
        gpts: list[_BaseGPT],
        allocation_function: str = "random",
        allocation_function_kwargs: Optional[dict] = None,
        allocation_function_weights: Optional[Sequence] = None,
        cache: Optional[Any] = None,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
    ):
        self.gpts = gpts

        if isinstance(allocation_function, str):
            allocation_function = ALLOCATION_FUNCTIONS[allocation_function]
        else:
            raise NotImplementedError(
                f"Cannot infer allocation function from type {type(allocation_function)}"
            )

        self.allocation_function = allocation_function
        self.allocation_function_kwargs = allocation_function_kwargs or {}

        if allocation_function_weights is not None:
            assert len(allocation_function_weights) == len(
                gpts
            ), "if provided, `allocation_function_weights` must be the same length as gpts"

        self.allocation_function_weights = allocation_function_weights

        super().__init__(
            cache=cache,
            max_parallel_calls=sum([gpt.max_parallel_calls for gpt in gpts]),
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )

    async def chat_completion(self, **kwargs) -> openai.ChatCompletion:
        gpt = await self.allocation_function(
            self.gpts,
            weights=self.allocation_function_weights,
            **self.allocation_function_kwargs,
        )

        return await gpt.chat_completion(**kwargs)


class LoadBalancedGPT(MultiLoadBalancedGPT):
    """
    We are continuing to support this for backward compatability reasons, but it is discouraged to use it.
    """

    def __init__(
        self,
        openai_api_key: str,
        azure_api_key: str,
        azure_api_base: str,
        azure_model_map: dict[str, str],
        cache: Optional[Any] = None,
        azure_openai_version: str = "2023-05-15",
        azure_openai_type: str = "azure",
        max_parallel_calls_openai: int = 5,
        max_parallel_calls_azure: int = 5,
        ratio_openai_to_azure: float = 0.25,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
    ):
        self.openai = ChatGPT(
            api_key=openai_api_key,
            cache=cache,
            max_parallel_calls=max_parallel_calls_openai,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )

        self.azure = AzureGPT(
            api_key=azure_api_key,
            azure_api_base=azure_api_base,
            azure_model_map=azure_model_map,
            azure_openai_version=azure_openai_version,
            azure_openai_type=azure_openai_type,
            cache=cache,
            max_parallel_calls=max_parallel_calls_azure,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )

        super().__init__(
            gpts=[self.openai, self.azure],
            cache=cache,
            allocation_function="random",
            allocation_function_weights=[
                ratio_openai_to_azure,
                1 - ratio_openai_to_azure,
            ],
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )
