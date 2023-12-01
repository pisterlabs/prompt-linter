from __future__ import annotations
import logging
from typing import Dict, List, Optional
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """ZhipuAI Embedding Model"""

    zhipuai_api_key: Optional[str] = None

    def _embed(self, text: str) -> List[float]:
        """
        send request

        :param text: input text

        :return: embeddings
        """
        try:
            resp = self.client.invoke(
                model="text_embedding",
                prompt=text
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if resp["code"] != 200:  # error
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (resp["code"], resp["msg"])
            )
        embeddings = resp["data"]["embedding"]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed search docs.

        :param texts: A list of text documents to embed.

        :return: List[List[float]]: A list of embeddings for each document in the input list.
            Each embedding is represented as a list of float values.
        """
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed query text.

        :param text: A text to embed.

        :return: List [float]: An embeddings list of input text, which is a list of floating-point values.
        """
        return self._embed(text)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validate whether zhipuai_api_key in the environment variables or configuration file are available or not.

        :param values: a dictionary containing configuration information, must include the fields of zhipuai_api_key

        :return: a dictionary containing configuration information. If zhipuai_api_key
            are not provided in the environment variables or configuration
            file, the original values will be returned; otherwise, values containing
            zhipuai_api_key will be returned.
        """
        values["zhipuai_api_key"] = get_from_dict_or_env(
            values,
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY"
        )

        try:
            import zhipuai
            zhipuai.api_key = values["zhipuai_api_key"]
            values["client"] = zhipuai.model_api

        except ImportError:
            raise ValueError(
                "Zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )
        return values
