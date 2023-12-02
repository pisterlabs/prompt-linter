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

import os
from enum import Enum
from typing import Dict
from typing import Optional

from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings

from .base import BaseModel


class ModelTypes(str, Enum):

    """
    Enum of available model types.
    """

    TextBison: str = "text-bison" #latest version
    TextBison001: str = "text-bison@001" 
    TextBison32k: str = "text-bison-32k" # latest version of text-bison with 32k token input

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
    Model Parameters.
    """

    class TextBison001:

        """
        Default parameters for text-bison model.
        """

        model_task: str = "text-bison@001"
        model_kwargs = {"temperature": 0.4, "max_tokens": 256, "max_retries": 3}
    
    class TextBison:

        """
        Default parameters for text-bison model in their lastest version
        """

        model_task: str = "text-bison"
        model_kwargs = {"temperature": 0.4, "max_tokens": 256, "max_retries": 3}
    
    class TextBison32k:

        """
        Default parameters for text-bison-32 model in their lastest version
        """

        model_task: str = "text-bison-32k"
        model_kwargs = {"temperature": 0.4, "max_tokens": 256, "max_retries": 3}


class GoogleVertexAILLM(BaseModel):

    """
    Google VertexAI LLM model.
    """

    def __init__(
        self,
        model_name: Optional[str] = "",
        model_params: Optional[Dict] = None,
        model_provider_token: Optional[str] = "",
        model_provider_project: Optional[str] = None,
    ) -> None:
        """
        Make predictions using a model from Google Vertex AI.
        It will use the os environment called GOOGLE_PROJECT_ID for instance the LLM
        """

        if not ModelTypes.has_value(model_name):
            raise ValueError(
                f"`model_name`={model_name} not in supported model names: "
                f"{[i.name for i in ModelTypes]}"
            )
        super(GoogleVertexAILLM, self).__init__()

        self._llm = VertexAI(
            model_name=model_name,
            project=model_provider_project or os.environ.get("GOOGLE_CLOUD_PROJECT_ID"),
        )

        self._embeddings = VertexAIEmbeddings()

        if not model_params:
            model_params = ModelParams[ModelTypes(model_name).name].value
        self.model_params = model_params
