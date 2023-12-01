from qdrant_client import QdrantClient
from qdrant_client.http import models

qdrant_client = QdrantClient(
    url="https://9c258883-7f0b-40bc-9830-f81a215c19f2.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="nkjC0SgmVYjT1vhHl8HZ3BHdnOQVC4IgWunB3MELFJSIbYYpfhgADA",
)


qdrant_client.recreate_collection(
    collection_name="{iching}",
    vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
)


import cohere
import qdrant_client

from qdrant_client.http.models import Batch

cohere_client = cohere.Client("kbeA3vuCuUxg2siLb7MPJN6AWE8d3R6wAxx7tghk")
qdrant_client = qdrant_client.QdrantClient()
qdrant_client.upsert(
    collection_name="iching",
    points=Batch(
        ids=[1],
        vectors=cohere_client.embed(
            model="large",
            texts=["The best vector database"],
        ).embeddings,
    )
)


