from typing import Callable, List
from langchain.schema import Document

from langchain.vectorstores.qdrant import Qdrant
from langchain.schema.embeddings import Embeddings

from generators.interfaces import EmbeddingModel

# Dependency inject embedding
def build_database(embedding: Embeddings | Callable[[str], List[float]], docs: List[Document], database_url:str="http://localhost", database_port:int=6333) -> Qdrant:
    embedding_model = embedding if isinstance(embedding, Embeddings) else EmbeddingModel(embedding)
    return Qdrant.from_documents(
        docs,
        embedding_model, # type: ignore
        url=database_url,
        port=database_port,
        #prefer_grpc=True,
        collection_name="Peter's useful notes",
        force_recreate=True,
        distance_func="Cosine",
    )