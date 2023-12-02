from typing import Optional

from ..clients.anthropic import AnthropicClient
from ..clients.base import ClientResponse
from ..clients.cohere import CohereClient
from ..clients.openai import OpenAIClient

__all__ = ["AnthropicAdapterMixin", "CohereAdapterMixin", "OpenAIAdapterMixin"]


class AnthropicAdapterMixin:
    def load_client(self, api_key: Optional[str] = None, **kwargs):
        client = AnthropicClient(api_key, **kwargs)
        client.set_target_api_endpoint("complete")
        return client

    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        if response.text is not None:
            response.text = response.text.strip()
        return response

    def prepare_input_content(self, content: str) -> dict:
        return {"prompt": content}


class CohereAdapterMixin:
    def load_client(self, api_key: Optional[str] = None, **kwargs):
        client = CohereClient(api_key, **kwargs)
        client.set_target_api_endpoint("complete")
        return client

    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        if response.text is not None:
            response.text = response.text.strip()
        return response

    def prepare_input_content(self, content: str) -> dict:
        return {"prompt": content}


class OpenAIAdapterMixin:
    def load_client(self, api_key: Optional[str] = None, **kwargs):
        client = OpenAIClient(api_key, **kwargs)
        client.set_target_api_endpoint("chat")
        return client

    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        if response.text is not None:
            response.text = response.text.strip()
        return response

    def prepare_input_content(self, content: str) -> dict:
        return {"messages": [{"role": "user", "content": content}]}
