import random
from http import HTTPStatus
from typing import Any, Dict, List, Mapping, Optional

import dashscope
from dashscope import Generation
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import root_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_result,
    stop_after_attempt,
)

from lib.config import read_config


def response_is_ok(response):
    if response.status_code == HTTPStatus.OK:
        return True
    return False


def return_last_message(retry_state):
    last_val = retry_state.outcome.result()
    result = {"output": {"text": f"Error: {last_val.code}: {last_val.message}"}}
    return result


@retry(
    retry=(retry_if_exception_type() | retry_if_not_result(response_is_ok)),
    stop=stop_after_attempt(3),
    retry_error_callback=return_last_message,
)
def get_reply(**kwargs):
    return Generation.call(**kwargs)


def get_from_dict_or_env(data, key, env_key):
    if key in data and data[key]:
        return data[key]
    else:
        config = read_config()
        if env_key in config and config[env_key]:
            return config[env_key]
        raise ValueError(
            f"Did not found {key} in provided dict and {env_key} in environment variables"
        )


class Alibaba(LLM):
    # TODO: maybe rewrite based on BaseLLM. Need to implement the more complex _generate method.
    model_name: Optional[str] = "qwen-v1"
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 100
    enable_search: Optional[bool] = False
    seed: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        return "alibaba"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:  # noqa: N805
        """Validate api key, python package exists."""
        dashscope_api_key = get_from_dict_or_env(
            values, "dashscope_api_key", "DASHSCOPE_API_KEY"
        )
        dashscope.api_key = dashscope_api_key

        if values["top_p"] is not None and not 0.0 <= values["top_p"] <= 1.0:
            raise ValueError("max_output_tokens must be between 0 and 1")

        if values["top_k"] is not None and not 1 <= values["top_k"] <= 100:
            raise ValueError("top_k must be between 1 and 100")

        return values

    def _call(
        self,
        prompt: str,
        messages: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        if messages is None:
            messages = []

        if self.seed is None:
            # FIXME: Alibaba's API support uint64
            # but I am not sure what's the max number I can generate with randint()
            seed = random.randint(0, 2**63)
            # seed = np.random.randint(2**64, dtype=np.uint64)  # this result in TypeError
        else:
            seed = self.seed

        result = get_reply.retry_with(
            stop=stop_after_attempt(
                3
            )  # TODO: set how many times to try as the class vars.
        )(
            model=self.model_name,
            prompt=prompt,
            messages=messages,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=seed,
            enable_search=self.enable_search,
        )
        return result["output"]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_search": self.enable_search,
        }
