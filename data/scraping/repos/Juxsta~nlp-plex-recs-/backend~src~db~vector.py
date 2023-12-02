import pinecone
from src.core.config import settings
from abc import ABC, abstractmethod
import openai as OpenAI
import logging
from typing import List
import math
logger = logging.getLogger(__name__)

def get_openai():
    OpenAI.api_key = settings.OPENAI_API_KEY
    return OpenAI

def create_embeddings(texts: List[str], batch_size = 2048) -> List[List[float]]:
    logger.info("Creating embeddings...")
    openai = get_openai()
    embeddings = []

    num_batches = math.ceil(len(texts) / batch_size)
    for i in range(num_batches):
        batch_texts = texts[i * batch_size: (i+1) * batch_size]
        embedding_response = openai.Embedding.create(input=batch_texts, model="text-embedding-ada-002")
        logger.debug(f"Embeddings created for batch {i+1} of {num_batches}.")
        embeddings.extend([data['embedding'] for data in embedding_response['data']])

    logger.info(f"Embeddings created for {len(texts)} texts.")
    return embeddings

class VectorDB(ABC):

    @abstractmethod
    def upsert(self, vectors, batch_size, **kwargs):
        pass

    @abstractmethod
    def query(self, query, top_k):
        pass

class PineconeDB(VectorDB):
    def __init__(self, index_name,**kwargs):
        """
        :param index_name: str - The name of the pinecone index.
        :param create_kwargs: Keyword arguments corresponding to the create_index method options.
            Expected keys can include but are not limited to:
            metric (str): ...
            shards (int): ...
            dimension (int): ...
        """
        pinecone.init(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, **kwargs)
        self.index = pinecone.Index(index_name=index_name)

    def upsert(self, vectors, batch_size, **kwargs):
        return self.index.upsert(vectors=vectors, batch_size=batch_size, **kwargs)

    def query(self, query, top_k, include_metadata=True, **kwargs):
        embedding = create_embeddings([query])
        query_result = self.index.query(queries=embedding, top_k=top_k, include_metadata=include_metadata, **kwargs)
        return query_result.results[0].matches

