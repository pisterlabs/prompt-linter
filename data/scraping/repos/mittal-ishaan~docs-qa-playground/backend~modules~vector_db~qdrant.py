import os
from qdrant_client import QdrantClient
from langchain.vectorstores.qdrant import Qdrant
from langchain.embeddings.base import Embeddings
from backend.modules.vector_db.base import BaseVectorDB
from backend.utils.base import VectorDBConfig


class QdrantVectorDB(BaseVectorDB):
    def __init__(self, config: VectorDBConfig, collection_name: str):
        self.url = config.url
        self.api_key = config.api_key
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(
            url=self.url, **({"api_key": self.api_key} if self.api_key else {})
        )

    def create_collection(self, embeddings: Embeddings):
        # No provision to create a empty collection
        return

    def upsert_documents(self, documents, embeddings: Embeddings):
        return Qdrant.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=self.collection_name,
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=True,
        )

    def delete_collection(self):
        return self.qdrant_client.delete_collection(collection_name="{collection_name}")

    def get_retriver(self, embeddings: Embeddings):
        return Qdrant(
            client=self.qdrant_client,
            embeddings=embeddings,
            collection_name=self.collection_name,
        ).as_retriever()
