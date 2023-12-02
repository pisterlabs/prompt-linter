from __future__ import annotations

from mimetypes import common_types
from typing import Dict, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types
from langchain.vectorstores.qdrant import Qdrant

from ..config.config import get_config
from .base import VectorStore
from .embedding.base import BaseEmbedding

DictFilter = Dict[str, Union[str, int, bool, dict, list]]
MetadataFilter = Union[DictFilter, common_types.Filter]


def create_qdrant_client(api_key: Optional[str] = None, url: Optional[str] = None, port: Optional[int] = None
                         ) -> QdrantClient:
    if api_key is None:
        qdrant_host_name = get_config("QDRANT_HOST_NAME") or "localhost"
        qdrant_port = int(get_config("QDRANT_PORT", default="6333"))
        qdrant_client = QdrantClient(host=qdrant_host_name, port=qdrant_port)
    else:
        qdrant_client = QdrantClient(api_key=api_key, url=url, port=port)
    return qdrant_client


class QdrantVectorStore(VectorStore):
    def __init__(
            self,
            client: QdrantClient,
            collection_name: str,
            embedding_model: BaseEmbedding,
    ):
        self.qdranr_client = client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.client = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embedding_model.client,
        )

    def get_index_stats(self) -> dict:
        """
        Returns:
            Stats or Information about a collection
        """
        collection_info = self.qdranr_client.get_collection(
            collection_name=self.collection_name)
        dimensions = collection_info.config.params.vectors.size
        vector_count = collection_info.vectors_count

        return {
            "dimensions": dimensions,
            "vector_count": vector_count,
        }
