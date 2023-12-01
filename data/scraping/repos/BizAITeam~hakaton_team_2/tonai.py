import json
from typing import List, Optional

import requests
from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests


class TonAI(LLM):
    
    def __init__(self, tonai_api_key, temperature=0.6,
                 service_name="chat", *args, **kwargs):
        super().__init__(*args, **kwargs)

        allowed_service_names = ["chat"]
        if service_name not in allowed_service_names:
            raise ValueError(f"Invalid service_name. Allowed values are: {', '.join(allowed_service_names)}")
        if not isinstance(tonai_api_key, str):
            raise ValueError("TonAI key must be a string")
        if not tonai_api_key:
            raise ValueError("TonAI key must not be empty")

        if not isinstance(self.endpoint_url, str) or not self.endpoint_url.startswith(("http://", "https://")):
            raise ValueError("Url must be a valid URL starting with 'http://' or 'https://'")
        if not isinstance(temperature, float) or temperature < 0:
            raise ValueError("Temperature must be a non-negative float")
        response = requests.get(self.endpoint_url, headers={"key": tonai_api_key})
        if response.ok:
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response: {e}") from None
        else:
            raise ValueError(f"Request failed with status code {response.status_code}")
        services = data['services']
        for item in services:
            if item['name'] == service_name:
                self.service_id = item['id']
                self.parameter_name = item['params']['required'][0]['name']
                break
        else:
            raise ValueError("No matching element found")
        self.temperature = temperature
        self.tonai_api_key = tonai_api_key

    endpoint_url: str = "https://tonai.tech/api/public/v1/services"
    """Endpoint URL to use."""
    temperature: float = 0
    """A non-negative float that tunes the degree of randomness in generation."""
    tonai_api_key: str = ""
    """This is the access key for accessing the neural network."""
    service_id: str = ""
    """This is the ID of the service provided by TonAI."""
    parameter_name: str = ""

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ton_ai"
    
    def _call(
        self,
        message: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        
        headers = {"key": self.tonai_api_key}

        payload = {
            'service_id': self.service_id,
            'temperature': self.temperature
            }

        payload[self.parameter_name] = json.dumps(message, ensure_ascii=False)

        try:
            response = requests.post(url=self.endpoint_url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}") from None

        if "error" in response_data:
            raise ValueError(f"Error raised by inference API: {response_data['error']}")

        answer = response_data['messages'][0]['text']
        return answer
