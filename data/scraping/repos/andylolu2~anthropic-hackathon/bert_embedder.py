from typing import Any, List

import torch
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel
from transformers import AutoModel, AutoTokenizer


class BertEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    model_name: str
    device: str = "cuda"
    max_length: int = 512
    tokenizer: Any = None
    model: Any = None

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        model_name = "michiyasunaga/BioLinkBERT-large"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    @torch.no_grad()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        outputs = self.model(**inputs)

        # LinkBERT uses a [CLS] token as the sequence embedding, which is the first token
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()
        return embedding

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
