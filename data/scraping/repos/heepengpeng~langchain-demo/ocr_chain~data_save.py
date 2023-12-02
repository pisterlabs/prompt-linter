import json
from abc import ABC
from typing import List, Dict, Any, Optional

import requests
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from pydantic import Extra


class DataSaveChain(Chain, ABC):
    """
        An example of a custom chain.
        """
    text: str = ""
    output_key: str = "response"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ["text"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return ocr_infer key.

        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        self.text = inputs["text"]
        return {self.output_key: self.http_request()}

    @property
    def _chain_type(self) -> str:
        return "save_data"

    def http_request(self):
        url = 'http://127.0.0.1:8000/api/save-contract/'
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, data=json.dumps(self.text))
        # 检查响应状态码
        if response.status_code == 200:
            print(response.text)
            return response.text
        else:
            print("Save contract error!")
            return ""
