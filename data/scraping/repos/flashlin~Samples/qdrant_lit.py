from abc import ABCMeta, abstractmethod

from langchain.schema import Document
from langchain.vectorstores import Qdrant
from torch.utils.data import Dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models


class VectorStoreBase(metaclass=ABCMeta):
    def __init__(self, llm_embedding):
        self.llm_embedding = llm_embedding
        self.embedding_dim = len(llm_embedding.get_embeddings('This is test text.'))

    @abstractmethod
    def create_collection(self, collection_name):
        pass

    @abstractmethod
    def upsert_docs(self, collection_name, docs):
        pass

    @abstractmethod
    def search(self, collection_name, query, k):
        pass


class QdrantVectorStore(VectorStoreBase):
    client: QdrantClient

    def open(self, url: str):
        self.client = QdrantClient(url)

    def create_collection(self, collection_name: str):
        collection = self.client.get_collection(collection_name)
        if collection is not None:
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                distance=models.Distance.COSINE,
                size=self.embedding_dim),
            optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
            hnsw_config=models.HnswConfigDiff(on_disk=True, m=16, ef_construct=100)
        )

    def recreate_collection(self, collection_name: str):
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                distance=models.Distance.COSINE,
                size=self.embedding_dim),
            optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
            hnsw_config=models.HnswConfigDiff(on_disk=True, m=16, ef_construct=100)
        )

    def get_all_collections(self):
        collections = self.client.get_collections()
        return collections

    def upsert_dataset(self, collection_name: str, dataset: Dataset):
        payloads = dataset.select_columns(["label_names", "text"]).to_pandas().to_dict(orient="records")
        self.client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=dataset["idx"],
                vectors=dataset["embedding"],
                payloads=payloads
            )
        )

    def upsert_docs(self, collection_name: str, docs: list[Document]):
        ids = []
        vectors = []
        payloads = []
        for idx, doc in enumerate(docs):
            embeddings = self.llm_embedding.get_embeddings(doc.page_content)
            ids.append(idx)
            vectors.append(embeddings)
            payload = {
                'page_content': doc.page_content,
                'source': doc.metadata['source']
            }
            payloads.append(payload)
        self.client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads  # [Payload(payload=point.payload) for point in docs_store]
            )
        )

    def get_store(self, collection_name):
        return Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=self.llm_embedding.embedding)

    def search(self, collection_name, query: str, k=3):
        query_embedding = self.llm_embedding.get_embeddings(query)
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k,
            append_payload=True,
        )
        return search_result
