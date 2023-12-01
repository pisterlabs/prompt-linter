# Importing necessary libraries and modules
from datetime import datetime
from langchain.schema import BaseRetriever, Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain.vectorstores import Qdrant
from rate_limiter import RateLimiter, SyncRateLimiter
from typing import (
    List,
    Optional,
    Tuple,
)

    
class QDrantVectorStoreRetriever(BaseRetriever):
    """Retriever that combines embedding similarity with conversation matching scores in retrieving values."""
    rate_limiter: RateLimiter
    rate_limiter_sync: SyncRateLimiter
    collection_name: str
    
    client: QdrantClient
    
    vectorstore: Qdrant
    """The vectorstore to store documents and determine salience."""

    extra_index_penalty: float = float(0.1)
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def _get_combined_score(
        self,
        query: str,
        document: Document,
        vector_relevance: Optional[float],
        extra_index: str = None
    ) -> float:
        """Return the combined score for a document."""
        score = 0
        if vector_relevance is not None:
            score += vector_relevance
        if extra_index is not None and extra_index != document.metadata.get("extra_index"):
            score -= self.extra_index_penalty
            if query == "":
                score = 0
        return score

    async def get_salient_docs(self, query: str, **kwargs) -> List[Tuple[Document, float]]:
        """Return documents that are salient to the query."""
        return await self.rate_limiter.execute(self.vectorstore.asimilarity_search_with_score, query, k=10, **kwargs)
    
    def _get_relevant_documents(self, *args, **kwargs):
        pass

    async def _aget_relevant_documents(
        self, query: str, **kwargs
    ) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.now().timestamp()
        extra_index = kwargs.pop("extra_index", None)
        user_filter = kwargs.pop("user_filter", None)
        if user_filter:
            kwargs.update({"filter": user_filter})
        docs_and_scores = await self.get_salient_docs(query, **kwargs)

        rescored_docs = []
        for doc, relevance in docs_and_scores:
            combined_score = self._get_combined_score(query, doc, relevance, extra_index)
            if combined_score != 0:  # Skip the document if the combined score is 0
                rescored_docs.append((doc, combined_score))
                # Ensure frequently accessed memories aren't forgotten
                doc.metadata["last_accessed_at"] = current_time

        # Sort by score and extract just the documents
        sorted_docs = [doc for doc, _ in sorted(rescored_docs, key=lambda x: x[1], reverse=True)]
        # Return just the list of Documents
        return sorted_docs

    def get_key_value_document(self, key, value) -> Document:
        """Get the key value from vectordb via scrolling."""
        filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key=key, 
                    match=rest.MatchValue(value=value), 
                )
            ]
        )
        record, _ = self.rate_limiter_sync.execute(self.client.scroll, collection_name=self.collection_name, scroll_filter=filter, limit = 1)
        if record is not None and len(record) > 0:
            return self.vectorstore._document_from_scored_point(
                record[0], self.vectorstore.content_payload_key, self.vectorstore.metadata_payload_key
            )
        else:
            return None