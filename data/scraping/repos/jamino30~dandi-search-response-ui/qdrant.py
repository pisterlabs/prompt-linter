from qdrant_client import QdrantClient as Qdrant
from qdrant_client.http.models import Distance, VectorParams, PointStruct, CollectionStatus
from qdrant_client.http.models import UpdateStatus
from concurrent.futures import ThreadPoolExecutor, wait

from .openai import OpenaiClient
from .embedding import EmbeddingClient

import os

def safe_int(value, default: int=0):
    try:
        return int(value)
    except Exception as e:
        return int(default)

class QdrantClient:

    def __init__(
        self,
        host: str=os.environ.get('QDRANT_HOST', "http://qdrant"),
        port: int=safe_int(os.environ.get('QDRANT_PORT'), 6333),
        api_key: str=os.environ.get('QDRANT_API_KEY', None),
    ) -> None:
        self.host = host
        self.port = port
        self.qdrant_client = Qdrant(
            location=self.host, 
            port=self.port,
            api_key=api_key,
        )


    def has_collection(self, collection_name: str):
        collection = self.qdrant_client.get_collection(collection_name=collection_name)
        return collection.dict()["status"] == CollectionStatus.GREEN


    def update_collection(self, collection_name: str, emb: list, vector_size: int):
        self.create_collection(collection_name=collection_name, vector_size=vector_size)
        self.add_points_to_collection(collection_name=collection_name, embeddings_objects=emb)
        

    def create_collection(self, collection_name: str, vector_size: int):
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.DOT),
        )   


    def add_points_to_collection(self, collection_name: str, embeddings_objects: list):
        def upsert_batch(points_batch):
            points_list = [PointStruct(**i) for i in points_batch]
            upsert_result = self.qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points_list,
            )
            assert upsert_result.status == UpdateStatus.COMPLETED

        with ThreadPoolExecutor() as executor:
            batch_size = 100
            batches = [embeddings_objects[i:i + batch_size] for i in range(0, len(embeddings_objects), batch_size)]
            futures = [executor.submit(upsert_batch, batch) for batch in batches]
            # Wait for all futures to complete
            wait(futures)
        print(f"All points added to collection {collection_name}")


    def get_collection_info(self, collection_name: str):
        return self.qdrant_client.get_collection(collection_name=collection_name).dict()


    def query_similar_items(self, collection_name: str, query: str, openai_client: OpenaiClient, emb_client: EmbeddingClient, top_k: int=10):
        if collection_name == "dandi_collection_ada002":
            query_vector = openai_client.get_embedding_simple(text=query)
        elif collection_name == "dandi_collection_emb":
            query_vector = emb_client.get_embedding_simple(text=query)
        else:
            raise ValueError("Invalid model selected.")

        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return search_result


    def query_from_user_input(
            self, 
            text: str, 
            collection_name: str, 
            openai_client: OpenaiClient,
            emb_client: EmbeddingClient,
            top_k: int=10,
        ):
        search_results = self.query_similar_items(
            query=text, 
            top_k=top_k, 
            collection_name=collection_name,
            openai_client=openai_client,
            emb_client=emb_client
        )
        results = dict()
        for sr in search_results:
            dandiset_id = f"DANDI:{sr.payload['dandiset_id']}/{sr.payload['dandiset_version']}"
            score = sr.score
            if dandiset_id not in results:
                results[dandiset_id] = score
            else:
                results[dandiset_id] += score
        return self.get_top_scores(results)


    def query_all_keywords(self, keywords: list, collection_name: str, top_k: int=10):
        results = dict()
        for keyword in keywords:
            search_results = self.query_similar_items(query=keyword, top_k=top_k, collection_name=collection_name)
            for sr in search_results:
                dandiset_id = f"DANDI:{sr.payload['dandiset_id']}/{sr.payload['dandiset_version']}"
                score = sr.score
                if dandiset_id not in results:
                    results[dandiset_id] = score
                else:
                    results[dandiset_id] += score
        return self.get_top_scores(results)


    def get_top_scores(self, dictionary: dict, N: int=None):
        sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        if not N:
            N = len(dictionary)
        top_scores = sorted_items[:min(N, len(dictionary))]
        return top_scores
