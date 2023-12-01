from __future__ import annotations

import abc

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore


class VectorStoreUnitOfWork(abc.ABC):
    """An abstract base class for vector store Unit of Work"""

    vector_store: VectorStore

    def __enter__(self) -> VectorStoreUnitOfWork:
        return self

    def __exit__(self, *args):
        pass

    @abc.abstractmethod
    def persist(self):
        raise NotImplementedError


# pylint: disable=too-few-public-methods
class LocalChromaUnitOfWork(VectorStoreUnitOfWork):
    """A Chroma-based Unit of Work"""

    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "langchain_store"

    def __init__(
        self,
        embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    ):
        self.vector_store = Chroma(
            embedding_function=embedding_function,
            collection_name=LocalChromaUnitOfWork.COLLECTION_NAME,
            persist_directory=LocalChromaUnitOfWork.PERSIST_DIRECTORY,
        )

    def persist(self):
        self.vector_store.persist()
