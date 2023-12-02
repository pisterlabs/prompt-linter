# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/10 17:15

"""Base interface for large language models to expose."""
import warnings
from typing import Any, Dict, Optional

from langchain.llms.openai import BaseOpenAI, OpenAIChat
from langchain.pydantic_v1 import root_validator



class AigcChat(OpenAIChat):...
    # nchar: Optional[int] = 1  # stream 字符
    # adapter_model: Optional[str] = "default"


class AigcLM(BaseOpenAI):
    # nchar: Optional[int] = 1 # stream 字符
    # adapter_model: Optional[str] = "default"

    """AigcLM models."""
    def __new__(cls, **data: Any) -> Union[AigcChat, BaseOpenAI]:  # type: ignore
        """Initialize the OpenAI object."""
        model_name = data.get("model_name", "")
        if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4") or model_name.lower().find('chat') != -1:
            warnings.warn(
                "You are trying to use a chat model. This way of initializing it is "
                "no longer supported. Instead, please use: "
                "`from langchain.chat_models import ChatOpenAI`"
            )
            return AigcChat(**data)
        return super().__new__(cls)

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._invocation_params}

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            import openlm

            values["client"] = openlm.Completion
        except ImportError:
            raise ImportError(
                "Could not import openlm python package. "
                "Please install it with `pip install openlm`."
            )
        if values["streaming"]:
            raise ValueError("Streaming not supported with openlm")
        return values

