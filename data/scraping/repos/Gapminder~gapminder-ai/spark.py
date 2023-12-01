"""Langchain Wrapper for iFlytek Spark
"""

from typing import Any, Dict, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import root_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from lib.app_singleton import app_logger as logger
from lib.config import read_config
from lib.llms.iflytek import SparkClient


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


class Spark(LLM):
    # TODO: maybe rewrite based on BaseLLM. Need to implement the more complex _generate method.
    client: Any
    iflytek_appid: str
    iflytek_api_key: str
    iflytek_api_secret: str
    temperature: Optional[float] = 0.5
    max_tokens: Optional[int] = 2048
    top_k: Optional[int] = 4

    @property
    def _llm_type(self) -> str:
        return "iflytek_spark"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:  # noqa: N805
        """Validate api key, python package exists."""
        iflytek_appid = get_from_dict_or_env(values, "iflytek_appid", "IFLYTEK_APPID")
        iflytek_api_key = get_from_dict_or_env(
            values, "iflytek_api_key", "IFLYTEK_API_KEY"
        )
        iflytek_api_secret = get_from_dict_or_env(
            values, "iflytek_api_secret", "IFLYTEK_API_SECRET"
        )

        values["client"] = SparkClient(
            iflytek_appid, iflytek_api_key, iflytek_api_secret
        )

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_k"] is not None and not 1 <= values["top_k"] <= 6:
            raise ValueError("top_k must be between 1 and 6")

        if values["max_tokens"] is not None and not 1 <= values["max_tokens"] <= 4096:
            raise ValueError("max_output_tokens must be between 1 and 4096")

        return values

    @retry(
        retry=(retry_if_exception_type()),
        stop=stop_after_attempt(3),
    )
    def generate_text_with_retry(self, prompt):
        return self.client.generate_text(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_k=self.top_k,
        )["text"]

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        output = self.generate_text_with_retry(prompt)
        logger.debug(f"Spark: {output}")
        return output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
        }
