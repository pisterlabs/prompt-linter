from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import VectorStore
from pydantic import BaseModel, Field


def _get_hours_passed(time: datetime, ref_time: datetime) -> float:
    """Get the hours passed between two datetime objects."""
    return (time - ref_time).total_seconds() / 3600


class MemoryStreamRetriever(BaseRetriever, BaseModel):
    """based on langchain's TimeWeightedVectorStoreRetriever,
    Retriever combining embedding similarity with recency.
    """

    get_timestamp: Callable[[], int]
    """The function to get the current timestamp, required."""

    vectorstore: VectorStore
    """The vectorstore to store documents and determine salience."""

    search_kwargs: dict = Field(default_factory=lambda: dict(k=100))
    """Keyword arguments to pass to the vectorstore similarity search."""

    # TODO: abstract as a queue
    memory_stream: List[Document] = Field(default_factory=list)
    """The memory_stream of documents to search through."""

    decay_rate: float = Field(default=0.01)
    """The exponential decay factor used as (1.0-decay_rate)**(hrs_passed)."""

    k: int = 4
    """The maximum number of documents to retrieve in a given call."""

    other_score_keys: List[str] = []
    """Other keys in the metadata to factor into the score, e.g. 'importance'."""

    default_salience: Optional[float] = None
    """The salience to assign memories not retrieved from the vector store.

    None assigns no salience to documents not fetched from the vector store.
    """

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Return documents that are relevant to the query."""
        ts = self.get_timestamp()
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self.k :]
        }

        # If a doc is considered salient, update the salience score
        docs_and_scores.update(self.get_salient_docs(query))
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, ts))
            for doc, relevance in docs_and_scores.values()
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []

        # Ensure frequently accessed memories aren't forgotten
        ts = self.get_timestamp()
        for doc, _ in rescored_docs[: self.k]:
            # TODO: Update vector store doc once `update` method is exposed.
            buffered_doc = self.memory_stream[doc.metadata["buffer_idx"]]
            buffered_doc.metadata["last_accessed_at"] = ts
            result.append(buffered_doc)
        return result

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Return documents that are relevant to the query."""
        raise NotImplementedError

    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: Optional[float],
        ts: int,
    ) -> float:
        """Return the combined score for a document."""
        score = (1.0 - self.decay_rate) ** (ts - document.metadata["last_accessed_at"])
        for key in self.other_score_keys:
            if key in document.metadata:
                score += document.metadata[key]
        if vector_relevance is not None:
            score += vector_relevance
        return score

    def get_salient_docs(self, query: str) -> Dict[int, Tuple[Document, float]]:
        """Return documents that are salient to the query."""
        docs_and_scores: List[Tuple[Document, float]]
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, **self.search_kwargs
        )
        results = {}
        for fetched_doc, relevance in docs_and_scores:
            buffer_idx = fetched_doc.metadata["buffer_idx"]
            doc = self.memory_stream[buffer_idx]
            results[buffer_idx] = (doc, relevance)
        # TODO: refactor this.
        #       those data have schema, but they ignore it and dump them into dict.
        #       use pydantic to parse them.
        return results

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        ts = kwargs.get("ts", self.get_timestamp())

        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        self._add_documents_to_memory_stream(dup_docs, ts)
        return self.vectorstore.add_documents(dup_docs, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Async add documents to vectorstore."""
        ts = kwargs.get("ts", self.get_timestamp())

        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        self._add_documents_to_memory_stream(dup_docs, ts)
        return await self.vectorstore.aadd_documents(dup_docs, **kwargs)

    def _add_documents_to_memory_stream(
        self,
        documents: List[Document],
        ts: int,
    ):
        """Add documents to vectorstore."""
        for i, doc in enumerate(documents):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = ts
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = ts
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(documents)
