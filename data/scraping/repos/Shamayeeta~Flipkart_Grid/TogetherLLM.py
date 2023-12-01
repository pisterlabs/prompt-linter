import together
import os
import logging
from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, Field, root_validator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

os.environ["TOGETHER_API_KEY"] = "de84cd526454f07fc441b1172ee5c55e78912c6c45c5374c100ac911b645f121" # API Key to access Together API Llama70b
# If API Key does not work, use either of the ones given below:
# 1. 9af1bb5fe28b10493f2e12b31afb38321a40db32eec7695bcdaf7775ac3544b6
# 2. 02d2cf4283560d97726b11926e6f986c744ea68b5f5d68770b8be97342175a8d

class TogetherLLM(LLM):

    model: str = "togethercomputer/llama-2-70b-chat" # model endpoint to use

    together_api_key: str = os.environ["TOGETHER_API_KEY"] # API Key to access Together API Llama70b

    temperature: float = 0.7 # sampling temperature

    max_tokens: int = 512 # The maximum number of tokens to generate in the completion

    class Config:
        extra = Extra.forbid

    # validates current API key
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str: # return type of LLM
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text
