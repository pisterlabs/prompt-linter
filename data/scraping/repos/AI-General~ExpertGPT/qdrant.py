from typing import Any, Coroutine, List, Optional, Callable, Tuple

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

class CustomQdrantVectorStore(Qdrant):
    """A custom vector store that uses the match_vectors table instead of the vectors table."""

    brain_id: str = "none"
    encoder = SentenceTransformer('all-MiniLM-L6-v2') 

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: OpenAIEmbeddings,
        encoder: SentenceTransformer = None,
        content_payload_key: str = "content",
        metadata_payload_key: str ="payload",
        brain_id: str = "none",
        # embedding_function: Optional[Callable] = None
    ):
        super().__init__(
            client=client, 
            collection_name=collection_name, 
            embeddings=embeddings,
            content_payload_key=content_payload_key, 
            metadata_payload_key=metadata_payload_key, 
        )
        self.brain_id = brain_id
        self.encoder = encoder

    def similarity_search(
        self,
        query: str,
        k: int = 6,
        threshold: float = 0.5,
        **kwargs: Any
    ) -> List[Document]:
        """
        self,
        query: str,
        k: int = 4,
        filter: Optional[MetadataFilter] = None,
        search_params: Optional[common_types.SearchParams] = None,
        offset: int = 0,
        score_threshold: Optional[float] = None,
        consistency: Optional[common_types.ReadConsistency] = None,
        **kwargs: Any,
        """
        # vectors = self._embeddings_function(query)
        query_vector=self.encoder.encode(query).tolist()
        # query_embedding = vectors

        res = self.client.search(
            collection_name=self.collection_name,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="brain_id",
                        match=models.MatchValue(
                            value=self.brain_id,
                        ),
                    )
                ]
            ),
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=False
            ),
            with_payload=["content", "brain_id"],
            query_vector=query_vector,
            limit=k,
        )

        match_result = [
            (
                Document(
                    metadata=search.payload,  # type: ignore
                    page_content=search.payload["content"]
                ),
                search.score,
            )
            for search in res
        ]

        documents = [doc for doc, _ in match_result]
        # print("####################################################################")
        # print(documents)

        return documents

    # @sync_call_fallback
    async def asimilarity_search(
        self,
        query: str,
        k: int = 6,
        threshold: float = 0.5,
        **kwargs: Any
    ) -> List[Document]:
        return self.similarity_search(query=query, k=k, threshold=threshold, **kwargs)
