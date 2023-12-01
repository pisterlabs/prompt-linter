"""Module for handling self-hosted LLama2 models"""

from typing import Any, List, Mapping, Optional
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema.output_parser import BaseOutputParser


class HostedLLM(LLM):
    """
    Class to define interaction with the hosted LLM at a specified URI
    """
    uri: str

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response = requests.get(self.uri,
                                params={"text" : prompt},timeout=600)
        if response.status_code == 200:
            return str(response.content)
        return f"Model Server is not Working due to error {response.status_code}"


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"uri": self.uri}

class CustomLlamaParser(BaseOutputParser[str]): # pylint: disable=R0903
    """Class to correctly parse model outputs"""

    def parse(self, text:str) -> str:
        """Parse the output of our LLM"""
        if text.startswith("Model Server is not Working due"):
            return text
        cleaned = str(text).split("[/INST]")
        return cleaned[1]

    @property
    def _type(self) -> str:
        return "custom_output_parser"
    