from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Extra, Field
from langchain.embeddings.base import Embeddings

from text2vec.sentence_model import SentenceModel


class MyEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from study_langchain.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any  #: :meta private:

    # model_name: str = DEFAULT_MODEL_NAME
    # """Model name to use."""
    # cache_folder: Optional[str] = None
    # """Path to store models.
    # Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    # model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # """Key word arguments to pass to the model."""
    # encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # """Key word arguments to pass when calling the `encode` method of the model."""

    def __init__(self, model_path, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        # try:
        #     import sentence_transformers
        #
        # except ImportError as exc:
        #     raise ImportError(
        #         "Could not import sentence_transformers python package. "
        #         "Please install it with `pip install sentence_transformers`."
        #     ) from exc

        # self.client = SentenceModel(model_name_or_path='shibing624/text2vec-base-chinese',device="cuda")
        self.client = SentenceModel(model_name_or_path=model_path)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        # embeddings = self.client.encode(texts, **self.encode_kwargs)
        embeddings = self.client.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        # embedding = self.client.encode(text, **self.encode_kwargs)
        embedding = self.client.encode(text)
        return embedding.tolist()
