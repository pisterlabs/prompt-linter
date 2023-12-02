from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable, List, Optional, Tuple
import numpy as np

from docarray import DocumentArray
from docarray import Document as DDocument
from langchain.embeddings.base import Embeddings

from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore


class DocArray(VectorStore):
    def __init__(
            self,
            index: DocumentArray,
            embedding_function: Callable):
        if not isinstance(index, DocumentArray):
            raise ValueError(f'client should be an instance of docarray.DocumentArray')
        self._index = index
        self._embedding_function = embedding_function

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> DocArray:
        embeddings = embedding.embed_documents(texts)
        docs = DocumentArray.empty(len(texts))
        docs.texts = texts
        docs.embeddings = np.array(embeddings)
        return cls(docs, embedding.embed_query)

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        docs = DocumentArray()
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        for i, text in enumerate(texts):
            embedding = self._embedding_function(text)
            metadata = metadatas[i] if metadatas else {}
            docs.append(DDocument(id=ids[i], embedding=np.array(embedding), tags=metadata))
        self._index.extend(docs)
        return ids

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4
    ) -> List[Tuple[Document, float]]:
        embedding = self._embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(embedding, k)
        return docs

    def similarity_search_by_vector(
            self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        docs = self.similarity_search_with_score_by_vector(embedding, k)
        return [d for d, _ in docs]

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        docs = self.similarity_search_with_score(query, k)
        return [d for d, _ in docs]

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4
    ) -> List[Tuple[Document, float]]:
        q = DocumentArray([DDocument(embedding=np.array(embedding))])
        q.match(self._index, metric='cosine', limit=k)
        docs = []
        for m in q[0].matches:
            docs.append((Document(page_content=m.text, metadata=m.tags), m.scores['cosine'].value))
        return docs

    @classmethod
    def from_da(cls, index: DocumentArray, embedding_function: Callable) -> DocArray:
        return cls(index=index, embedding_function=embedding_function)
