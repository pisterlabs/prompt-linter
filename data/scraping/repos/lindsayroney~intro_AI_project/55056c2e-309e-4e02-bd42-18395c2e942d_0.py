from abc import ABC, abstractmethod
import redis
import openai
import numpy as np
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

class EmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str):
        pass

class OpenAIEmbeddingModel(EmbeddingModel):
    def create_embedding(self, text: str):
        embedding = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        vector = embedding["data"][0]["embedding"]
        vector = np.array(vector).astype(np.float32).tobytes()
        return vector

class EmbeddingStorage(ABC):
    @abstractmethod
    def store(self, key: str, vector):
        pass

    @abstractmethod
    def retrieve(self, key: str):
        pass

    @abstractmethod
    def search(self, vector, top_k=5):
        pass

class RedisEmbeddingStorage(EmbeddingStorage):
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host=host, port=port, db=db, encoding='utf-8', decode_responses=True)
        self.SCHEMA = [
            TextField("url"),
            VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"}),
        ]
        try:
            self.r.ft("posts").create_index(fields=self.SCHEMA, definition=IndexDefinition(prefix=["post:"], index_type=IndexType.HASH))
        except Exception as e:
            print("Index already exists")

    def store(self, key: str, vector):
        post_hash = {
            "url": key,
            "embedding": vector
        }
        self.r.hset(name=f"post:{key}", mapping=post_hash)

    def retrieve(self, key: str):
        return self.r.hget(name=f"post:{key}", key="embedding")

    def search(self, vector, top_k=5):
        base_query = f"*=>[KNN {top_k} @embedding $vector AS vector_score]"
        query = Query(base_query).return_fields("url", "vector_score").sort_by("vector_score").dialect(2)
        try:
            results = self.r.ft("posts").search(query, query_params={"vector": vector})
        except Exception as e:
            print("Error calling Redis search: ", e)
            return None
        return results
