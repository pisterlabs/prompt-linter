import cohere
from fastchain.embedding.base import BaseEmbedding
import numpy as np
from enum import Enum
from typing import Any, Dict, List
from fastchain.document.base import Document
import os

EMB_TYPE = np.ndarray
DEFAULT_EMBED_BATCH_SIZE = 10
apikey = os.environ.get("COHERE_API_KEY")   
co = cohere.Client(apikey)

class CohereEmbeddingModels(Enum):
    """Cohere embedding models."""
    EMBED_ENGLISH = "embed-english-v2.0"
    EMBED_ENGLISH_LIGHT = "embed-english-light-v2.0"
    EMBED_MULTILINGUAL = "embed-multilingual-v2.0"

def get_cohere_engine(model: str) -> CohereEmbeddingModels:
    """Get Cohere engine."""
    if model not in CohereEmbeddingModels._value2member_map_:
        raise ValueError(f"Invalid model: {model}")
    return CohereEmbeddingModels(model)

class CohereEmbedding(BaseEmbedding):
    """Cohere class for embeddings."""
    
    def __init__(self, model: str = CohereEmbeddingModels.EMBED_ENGLISH.value, **kwargs: Any) -> None:
        super().__init__()
        self.cohere_model = get_cohere_engine(model)
        
    def _get_embedding(self, content: str) -> List[float]:
        response = co.embed(texts=[content], model=self.cohere_model.value)
        return response

    async def _aget_embedding(self, content: str) -> List[float]:
        # Cohere's client doesn't have an async method, so we can reuse the synchronous one
        return await self._get_embedding(content)

    def queue_document_for_embedding(self, document: Document) -> None:
        self._document_queue.append(document)

    def _process_documents(self, async_mode: bool = False) -> Dict[str, Dict[str, EMB_TYPE]]:
        result = {}
        for document in self._document_queue:
            doc_embeddings = {}
            for page in document.pages or []:
                for chunk in page.chunks or []:
                    emb_func = self.aget_chunk_embedding if async_mode else self.get_chunk_embedding
                    doc_embeddings[chunk._id] = emb_func(chunk)
            for chunk in document.chunks or []:
                emb_func = self.aget_chunk_embedding if async_mode else self.get_chunk_embedding
                doc_embeddings[chunk._id] = emb_func(chunk)
            result[document._id] = doc_embeddings
        self._document_queue.clear()
        return result

    def get_queued_document_embeddings(self) -> Dict[str, Dict[str, EMB_TYPE]]:
        return self._process_documents()

    async def aget_queued_document_embeddings(self) -> Dict[str, Dict[str, EMB_TYPE]]:
        return await self._process_documents(async_mode=True)