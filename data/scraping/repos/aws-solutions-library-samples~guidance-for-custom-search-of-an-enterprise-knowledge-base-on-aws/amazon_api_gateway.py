from typing import Any, Dict, List, Mapping, Optional

import requests
from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class ContentHandlerAmazonAPIGateway:
    """Adapter class to prepare the inputs from Langchain to a format
    that LLM model expects. Also, provides helper function to extract
    the generated text from the model response."""

    @classmethod
    def transform_input(
        cls, prompt: str, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"inputs": prompt, "parameters": model_kwargs}

    @classmethod
    def transform_output(cls, response: Any) -> str:
        return response.json()[0]["generated_text"]


class AmazonAPIGateway(LLM):
    """Wrapper around custom Amazon API Gateway"""

    api_url: str
    """API Gateway URL"""

    headers: Optional[Dict] = None
    """API Gateway HTTP Headers to send, e.g. for authentication"""

    model_kwargs: Optional[Dict] = None
    """Key word arguments to pass to the model."""

    content_handler: ContentHandlerAmazonAPIGateway = ContentHandlerAmazonAPIGateway()
    """The content handler class that provides an input and
    output transform functions to handle formats between LLM
    and the endpoint.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_name": self.api_url},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "amazon_api_gateway"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Amazon API Gateway model.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = se("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}
        payload = self.content_handler.transform_input(prompt, _model_kwargs)

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
            )
            text = self.content_handler.transform_output(response)

        except Exception as error:
            raise ValueError(f"Error raised by the service: {error}")

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text
