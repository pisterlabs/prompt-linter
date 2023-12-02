from qdrant_client import QdrantClient as Qdrant
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import UpdateStatus
from concurrent.futures import ThreadPoolExecutor, wait

from ..core.settings import settings
from .openai import OpenaiClient


class QdrantClient:

    def __init__(
        self,
        host: str=settings.QDRANT_HOST,
        port: int=settings.QDRANT_PORT,
        vector_size: int=settings.QDRANT_VECTOR_SIZE,
        api_key: str=settings.QDRANT_API_KEY,
    ) -> None:
        self.host = host
        self.port = port
        self.vector_size = vector_size
        self.qdrant_client = Qdrant(
            location=self.host, 
            port=self.port,
            api_key=api_key,
        )
        self.openai_client = OpenaiClient()


    def create_collection(self, collection_name: str):
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.DOT),
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


    def query_similar_items(self, collection_name: str, query: str, top_k: int=10):
        query_vector = self.openai_client.get_embedding_simple(query)
        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return search_result


    def query_from_user_input(self, text: str, collection_name: str, top_k: int=10):
        search_results = self.query_similar_items(query=text, top_k=top_k, collection_name=collection_name)
        results = dict()
        for sr in search_results:
            dandiset_id = sr.payload["dandiset_id"]
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
                dandiset_id = sr.payload["dandiset_id"]
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
