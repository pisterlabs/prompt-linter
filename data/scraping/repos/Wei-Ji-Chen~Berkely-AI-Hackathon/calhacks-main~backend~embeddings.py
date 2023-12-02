from typing import Any, Dict, Union
import pinecone
import openai

from backend.config import PINECONE_API_KEY

class Pinecone:
    def __init__(
            self,
            index: str,
            environment: str = "asia-southeast1-gcp-free",
            embeddings_model: str = "text-embedding-ada-002",
    ):
        pinecone.init(api_key=PINECONE_API_KEY, environment=environment)

        self.environment = environment
        self.embeddings_model = embeddings_model
        self.index = pinecone.Index(index)

    def get_similar(self, text: str, count: int = 2, filter: Dict[str, Any] = {}):
        embeddings = openai.Embedding.create(
            input=text,
            model=self.embeddings_model,
        )
        embeddings = embeddings["data"][0]["embedding"]  # type: ignore

        result = self.index.query(
            vector=embeddings,
            top_k=count,
            include_values=True,
            filter=filter,
        )

        return [item["id"] for item in result["matches"]]

    def store(self, id: Union[str, int], text: str, metadata: Dict[str, Any]):
        embeddings = openai.Embedding.create(
            input=text,
            model=self.embeddings_model,
        )
        embeddings = embeddings["data"][0]["embedding"]  # type: ignore

        self.index.upsert([(id, embeddings, metadata)])
