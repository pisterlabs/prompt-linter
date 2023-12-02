"""Wrapper around HuggingFace embedding models."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field

from langchain.embeddings.base import Embeddings

DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"


class Ct2BertEmbeddings(BaseModel, Embeddings):
    """Wrapper around CTranslate2 BERT embedding models.
    https://opennmt.net/CTranslate2/guides/transformers.html#bert

    To use, you should have the ``ctranslate2`` python package installed.

    Example:
        .. code-block:: python

            from custom.embeddings.ctranslate2 import Ct2BertEmbeddings

            model_name = "intfloat/e5-large-v2"
            model_kwargs = {'device': 'cpu', 'compute_type':'int8'}
            encode_kwargs = {'batch_size': 32, 'convert_to_numpy': True, 'normalize_embeddings': True}
            embeddings = Ct2BertEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when calling the `encode` method of the model."""

    def __init__(self, **kwargs: Any):
        """Initialize the CTranslate2 BERT Encoder."""
        super().__init__(**kwargs)
        try:
            from hf_hub_ctranslate2 import CT2SentenceTransformer

        except ImportError as exc:
            raise ImportError(
                "Could not import hf_hub_ctranslate2 python package. "
                "Please install it with `pip install hf_hub_ctranslate2`."
            ) from exc

        self.client = CT2SentenceTransformer(
            model_name_or_path=self.model_name,
            **self.model_kwargs
        )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a CTranslate2 BERT model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client.encode(texts, **self.encode_kwargs)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a CTranslate2 BERT model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embeddings = self.embed_documents([text])
        return embeddings[0]
