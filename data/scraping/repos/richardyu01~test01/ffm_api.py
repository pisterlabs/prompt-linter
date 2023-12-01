'''Wrapper LLM APIs.'''
from typing import Any, Dict, List, Mapping, Optional
from langchain.llms.base import LLM
import requests
from pydantic import Field

class FormosaFoundationModel(LLM):
    endpoint_url: str = ''
    max_new_tokens: int = 20
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    frequence_penalty: float = 1.0
    model_kwargs: Dict[str, Any] = dict()
    ffm_api_key: Optional[str] = None
    model: str=''

    @property
    def _llm_type(self) -> str:
        return 'FormosaFoundationModel'
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        params = self._invocation_params
        parameter_payload = {"inputs": prompt, "parameters": params, "model": self.model}

        # HTTP headers for authorization
        headers = {
            'X-API-KEY': self.ffm_api_key,
            'Content-Type': 'application/json',
        }

        # send request
        try:
            response = requests.post(
                self.endpoint_url, headers=headers, json=parameter_payload
            )
            if response.status_code != 200:
                return f'http error: {response.reason}'

        except requests.exceptions.RequestException as e:  # This is the correct syntax
            raise ValueError(f"Error raised by inference endpoint: {e}")

        generated_text = response.json()

        if generated_text.get('detail') is not None:
            msg = generated_text['detail']
            raise ValueError(
                f'Error raised by inference API: {msg}'
            )
        
        if generated_text.get('generated_text') is None:
            return 'Response format error'

        text = generated_text['generated_text'].lstrip('\n')

        return text
        
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling FFM API."""
        normal_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "frequence_penalty": self.frequence_penalty,
            "top_k": self.top_k,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        '''Get the identifying parameters.'''
        
        return {
            **{"endpoint_url": self.endpoint_url},
            **self._default_params
        }

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return self._default_params
