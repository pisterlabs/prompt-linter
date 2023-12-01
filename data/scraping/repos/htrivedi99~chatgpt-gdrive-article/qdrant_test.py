import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct, CollectionStatus, UpdateStatus
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http import models
from typing import List
import openai
from openai.embeddings_utils import get_embedding
openai.api_key = "YOUR-OPENAI-API-KEY"


class QdrantVectorStore:

    def __init__(self,
                 host: str = "localhost",
                 port: int = 6333,
                 db_path: str = "/Users/het/qdrant/qdrant_storage",
                 collection_name: str = "test_collection",
                 vector_size: int = 1536,
                 vector_distance=Distance.COSINE
                 ):

        self.client = QdrantClient(
            url=host,
            port=port,
            path=db_path
        )
        self.collection_name = collection_name

        try:
            collection_info = self.client.get_collection(collection_name=collection_name)
        except Exception as e:
            print("Collection does not exist, creating collection now")
            self.set_up_collection(collection_name,  vector_size, vector_distance)

    def set_up_collection(self, collection_name: str, vector_size: int, vector_distance: str):

        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=vector_distance)
        )

        collection_info = self.client.get_collection(collection_name=collection_name)

    def upsert_data(self, data: List[dict]):
        points = []
        for item in data:
            text = item.get("text")

            text_vector = get_embedding(text, engine="text-embedding-ada-002")
            text_id = str(uuid.uuid4())
            point = PointStruct(id=text_id, vector=text_vector, payload=item)
            points.append(point)

        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points)

        if operation_info.status == UpdateStatus.COMPLETED:
            print("Data inserted successfully!")
        else:
            print("Failed to insert data")

    def search(self, input_query: str, limit: int = 3):
        input_vector = get_embedding(input_query, engine="text-embedding-ada-002")
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=input_vector,
            limit=limit
        )

        result = []
        for item in search_result:
            similarity_score = item.score
            payload = item.payload
            data = {"id": item.id, "similarity_score": similarity_score, "text": payload.get("text")}
            result.append(data)

        return result


    def search_with_filter(self, input_query: str, filter: dict, limit: int = 3):
        input_vector = get_embedding(input_query, engine="text-embedding-ada-002")
        filter_list = []
        for key, value in filter.items():
            filter_list.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=input_vector,
            query_filter=Filter(
                must=filter_list
            ),
            limit=limit
        )

        result = []
        for item in search_result:
            similarity_score = item.score
            payload = item.payload
            data = {"id": item.id, "similarity_score": similarity_score, "quote": payload.get("quote"),
                    "person": payload.get("person")}
            result.append(data)

        return result

    def delete(self, text_ids: list):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=text_ids,
            )
        )

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name=collection_name)
        print("collection deleted")

    def get_collection(self, collection_name: str):
        return self.client.get_collection(collection_name=collection_name)

