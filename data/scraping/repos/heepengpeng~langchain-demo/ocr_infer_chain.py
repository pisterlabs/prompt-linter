from __future__ import annotations

import os.path
from abc import ABC
from typing import Any, Dict, List, Optional

import requests
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from pydantic import Extra


def ocr_agreement(contract_path) -> str:
    with open(contract_path, 'rb') as f:
        image_data = f.read()
        files = {'image': ('image.jpg', image_data, 'image/jpeg')}
        response = requests.post('http://127.0.0.1:8000/api/ocr-image/', files=files)
        if response.status_code == 200:
            return response.json()['ocr_text']
        else:
            return ""


class OCRInferChain(Chain, ABC):
    """
        An example of a custom chain.
        """

    contract_path: str = ""
    output_key: str = ""  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ["contract_path"]

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
        self.contract_path = inputs["contract_path"]
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp/")

        contract_text = ocr_agreement(self.contract_path)
        with open('./tmp/contract.txt', "w") as f:
            f.write(contract_text)
        return {self.output_key: contract_text}

    @property
    def _chain_type(self) -> str:
        return "ocr_infer"
