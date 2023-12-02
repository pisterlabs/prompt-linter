import os

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API")
)


def get_or_create_collection(name: str, collection_size: int) -> types.CollectionInfo:
    if (has_collection(name)):
        collection = qdrant_client.get_collection(collection_name=name)
    else:
        collection = qdrant_client.recreate_collection(collection_name=name,
                                                       vectors_config=models.VectorParams(
                                                           size=collection_size,
                                                           distance=models.Distance.COSINE,
                                                       )
                                                       )

    return collection


def has_collection(name: str) -> bool:
    for descr in qdrant_client.get_collections().collections:
        if descr.name == name:
            return True


def get_vector_store(collection_name: str,
                     embeddings: HuggingFaceInstructEmbeddings
                     ) -> Qdrant:
    return Qdrant(client=qdrant_client,
                  collection_name=collection_name,
                  embeddings=embeddings
                  )


def store_to_db(bucket, path, data):
    pass
