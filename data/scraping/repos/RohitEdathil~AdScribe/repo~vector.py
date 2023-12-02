from abc import ABC, abstractmethod
from models.product import Product
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


class VectorRepository(ABC):
    vectorstore: VectorStore

    def __init__(self, embedding: Embeddings) -> None:
        self.embedding = embedding

    @abstractmethod
    def ingest(self, product: Product) -> None:
        """Ingest product into vector store"""
        pass
