#!/usr/bin/python3

#  Copyright (c) 2023 Paradigma Digital S.L.

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from enum import Enum
from typing import Optional

from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings

from .base import BaseModel


class ModelTypes(str, Enum):

    """
    Enum of available model types.
    """

    FlanT5Xxl: str = "google/flan-t5-xxl"

    @classmethod
    def has_value(
        cls,
        value: str,
    ) -> bool:
        """
        Checks if the value is in the enum or not.
        """

        return value in cls._value2member_map_


class ModelParams(Enum):

    """
    Model Parameters Enum
    """

    class FlanT5Xxl:

        """
        Flan-t5-xxl default params
        """

        model_task = "text2text-generation"
        model_kwargs = {"temperature": 0.9, "max_length": 64}


class HFHubApiLLM(BaseModel):

    """
    HuggingFace API call.
    """

    def __init__(
        self,
        model_name: Optional[str] = "",
        model_params: Optional[ModelParams] = None,
        model_provider_token: Optional[str] = "",
    ) -> None:
        """
        Make predictions using a model from HuggingFace using the API.
        """

        if ModelTypes.has_value(model_name):
            model_params = ModelParams[ModelTypes(model_name).name].value
        elif not model_params:
            raise ValueError(
                f"`model_name`={model_name} not in supported model names: "
                f"{[i.value for i in ModelTypes]}"
            )
        elif not hasattr(model_params, "model_kwargs"):
            raise ValueError("invalid model_params")

        super(HFHubApiLLM, self).__init__()

        self._llm = HuggingFaceHub(
            repo_id=model_name,
            huggingfacehub_api_token=model_provider_token,
            model_kwargs=model_params.model_kwargs,
        )

        self._embeddings = HuggingFaceHubEmbeddings(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=model_provider_token,
        )
