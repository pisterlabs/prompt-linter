from typing import Any, Dict, Optional

from langchain.pydantic_v1 import Field
from langchain.embeddings import HuggingFaceEmbeddings
from semantic_ai.embeddings.base import BaseEmbeddings

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class HFEmbeddings(BaseEmbeddings):
    """HuggingFace sentence_transformers embedding models.

        To use, you should have the ``sentence_transformers`` python package installed.

        Example:
            . code-block:: python

                from semantic.embeddings.huggingface import HFEmbeddings

                model_name = "sentence-transformers/all-mpnet-base-v2"
                model_kwargs = {'device': 'cpu'}
                encode_kwargs = {'normalize_embeddings': False}
                hf = HFEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
        """

    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def embed(self, **kwargs):
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            cache_folder=self.cache_folder,
            multi_process=self.multi_process,
            **kwargs
        )
        return embeddings
