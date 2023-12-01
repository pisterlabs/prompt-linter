from typing import Any, Optional, List, Mapping, Dict
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from config import Config


class Educhat(LLM):
    _url: str = Config().get("EDUCHAT_SECRET_URL")
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    prompt: Optional[str]


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.top_p = kwargs.get('top_p', 0.7)
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 2048)

    @property
    def _llm_type(self) -> str:
        return "educhat"

    def _call(self, prompt, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None,
              *args, **kwargs):
        headers = {'Content-Type': 'application/json'}
        messages = [{'role': 'user', 'content': prompt}]
        json = {
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p
        }
        response = requests.post(self._url, headers=headers, json=json)
        response.raise_for_status()
        return response.json()['response']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "top_p": self.top_p,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
