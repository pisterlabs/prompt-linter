import os
import uuid

from dotenv import load_dotenv, find_dotenv
from icecream import ic
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct, CollectionStatus, UpdateStatus
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http import models
from typing import List

import openai
from openai.embeddings_utils import get_embedding

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# source: https://www.brainbyte.io/vector-search-using-openai-embeddings-with-qdrant/

# possible other tutorials:
# https://dylancastillo.co/ai-search-engine-fastapi-qdrant-chatgpt/

# this one uses langchain
# https://cookbook.openai.com/examples/vector_databases/qdrant/qa_with_langchain_qdrant_and_openai

class QdrantVectorStoreSample:

    def __init__(self,
                 host: str = None,
                 port: int = None,
                 # db_path: str = "/path/to/db/qdrant_storage",
                 collection_name: str = None,
                 vector_size: int = None,
                 vector_distance=Distance.COSINE
                 ):

        self.client = QdrantClient(
            url=host,
            port=port,
            # path=db_path
        )
        self.collection_name = collection_name

        try:
            collection_info = self.client.get_collection(collection_name=collection_name)
        except Exception as e:
            ic("Collection does not exist, creating collection now")
            self.set_up_collection(collection_name, vector_size, vector_distance)

    def set_up_collection(self, collection_name: str, vector_size: int, vector_distance: str):

        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=vector_distance)
        )

        collection_info = self.client.get_collection(collection_name=collection_name)

    def upsert_data(self, data: List[dict]):
        points = []
        for item in data:
            quote = item.get("quote")
            person = item.get("person")

            text_vector = get_embedding(quote, engine="text-embedding-ada-002")
            text_id = str(uuid.uuid4())
            payload = {"quote": quote, "person": person}
            point = PointStruct(id=text_id, vector=text_vector, payload=payload)
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
            data = {"id": item.id, "similarity_score": similarity_score, "quote": payload.get("quote"),
                    "person": payload.get("person")}
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


if __name__ == "__main__":
    # Your Qdrant host and port
    qdrant_host = 'localhost'
    qdrant_port = 6333

    # Store the data in Qdrant
    qdrant_collection_name = 'testing'
    ai_vector_size = 1536

    vector_db = QdrantVectorStoreSample(host=qdrant_host,
                                        port=qdrant_port,
                                        vector_size=ai_vector_size,
                                        collection_name=qdrant_collection_name)
    famous_quotes = [
        {"quote": "A rose by any other name would smell as sweet.", "person": "William Shakespeare"},
        {"quote": "All that glitters is not gold.", "person": "William Shakespeare"},
        {"quote": "Ask not what your country can do for you; ask what you can do for your country.",
         "person": "John Kennedy"},
        {"quote": "Genius is one percent inspiration and ninety-nine percent perspiration.", "person": "Thomas Edison"},
        {"quote": "He travels the fastest who travels alone.", "person": "Rudyard Kipling"},
        {"quote": "Houston, we have a problem.", "person": "Jim Lovell"},
        {"quote": "That’s one small step for a man, a giant leap for mankind.", "person": "Neil Armstrong"}
    ]

    response = vector_db.client.count(collection_name=vector_db.collection_name)
    ic(f"count:{response.count}")

    ic('----------------')
    if response.count == 0:
        ic('------upsert_data----------')
        vector_db.upsert_data(famous_quotes)
    else:
        ic('------db full----------')

    ic('----------------')
    query = 'Genius'
    result = vector_db.search(query, limit=3)
    ic(f"{query}:{result}")

    result = vector_db.search("gold rush")
    ic(f"gold rush:{result}")

    ic(f"gold rush:{result}")
    ic('----------------')
    results = vector_db.search_with_filter("gold rush", filter={"person": "William Shakespeare"})
    ic(f"gold rush + person:{results}")

    ic('----------------')

    # delete_result = vector_db.delete(text_ids=["ea44cfaa-c14e-40d8-8f99-931e564d8539"])
    # ic(f'delete {delete_result}')
    # ic('----------------')
