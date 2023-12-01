import logging
from typing import Any, List, Mapping, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class SparkLLM(LLM):
    """Define the custom LLM wrapper for Xunfei SparkLLM to get support of LangChain
    """基于langchain的LLM基类定制讯飞星火大模型类提供对星火大模型的调用


    endpoint_url: str = "http://127.0.0.1:8000/qa?"
    """Endpoint URL to use.此URL指向部署的调用星火大模型的FastAPI接口地址"""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""
    #max_token: int = 4000
    """Max token allowed to pass to the model.在真实应用中考虑启用"""
    #temperature: float = 0.75
    """LLM model temperature from 0 to 10.在真实应用中考虑启用"""
    #history: List[List] = []
    """History of the conversation.在真实应用中可以考虑是否启用"""
    #top_p: float = 0.85
    """Top P for nucleus sampling from 0 to 1.在真实应用中考虑启用"""
    #with_history: bool = False
    """Whether to use history or not.在真实应用中考虑启用"""

    @property
    def _llm_type(self) -> str:
        return "SparkLLM"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.endpoint_url},
            **{"model_kwargs": _model_kwargs},
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        payload={"query":prompt}

        # call api
        try:
            response = requests.get(self.endpoint_url, params=payload)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        logger.debug(f"SparkLLM response: {response}")

        if response.status_code != 200:
            raise ValueError(f"Failed with response: {response}")
        
        text = response.content
        return text
