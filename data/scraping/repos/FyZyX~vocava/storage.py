import uuid

import chromadb
import cohere
import numpy
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# compare them
def calculate_similarity(actual, expected, api_key, model='embed-multilingual-v2.0'):
    co = cohere.Client(api_key)
    docs = [actual.strip().strip(".").lower(), expected.strip().strip(".").lower()]
    actual_embed, expected_embed = co.embed(docs, model=model).embeddings
    norm_product = numpy.linalg.norm(actual_embed) * numpy.linalg.norm(expected_embed)
    return numpy.dot(actual_embed, expected_embed) / norm_product


class Document:
    def __init__(self, content: str, metadata: dict):
        self._id = str(uuid.uuid4())
        self._content = content
        self._metadata = metadata

    def id(self):
        return self._id

    def content(self):
        return self._content

    def metadata(self):
        return self._metadata


class VectorStore:
    _db = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=".chromadb",
    ))

    def __init__(self, cohere_api_key):
        self._cohere_api_key = cohere_api_key
        self._embedding_function = embedding_functions.CohereEmbeddingFunction(
            api_key=self._cohere_api_key,
            model_name="embed-multilingual-v2.0",
        )
        self._collection: chromadb.api.Collection = self._db.get_or_create_collection(
            name="vocava",
            embedding_function=self._embedding_function,
        )

    def save(self, *documents: Document) -> bool:
        if not self._collection:
            raise ValueError("Must call connect before querying.")

        self._collection.add(
            ids=[doc.id() for doc in documents],
            documents=[doc.content() for doc in documents],
            metadatas=[doc.metadata() for doc in documents],
        )
        try:
            self._db.persist()
        except RuntimeError:
            return False
        return True

    def query_by_metadata(self, **metadata):
        return self._collection.get(
            where=metadata
        )
