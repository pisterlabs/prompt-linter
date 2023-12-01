import json
import os
from typing import Iterable, List

import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams

from markdown_search.config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, DATA_DIR
from markdown_search.encoder import OpenAIEncoder

BATCH_SIZE = 32
EMBEDDING_SIZE = 1536
ENCODER_MODEL = "text-embedding-ada-002"

encoder = OpenAIEncoder(ENCODER_MODEL)


def iter_batch(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_records(filename: str) -> Iterable[dict]:
    with open(filename, 'r') as f:
        for line in f:
            json_obj = json.loads(line)
            yield {
                "id": f"{json_obj.get('file')}-{json_obj.get('url')}",
                "text": json_obj.get("context"),
                "metadata": {
                    "document_id": json_obj.get("file"),
                    "url": json_obj.get("url"),
                },
                "created_at": None,
            }


def read_text_records(filename: str, reader=read_records) -> Iterable[str]:
    for record in reader(filename):
        yield record['text']


def encode_texts(texts: Iterable[str]) -> Iterable[List[float]]:
    for batch in iter_batch(texts, BATCH_SIZE):
        yield from encoder.encode_batch(batch)


def upload(records_path):
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY,
                                 prefer_grpc=True)

    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )

    payloads = read_records(records_path)
    vectors = encode_texts(tqdm.tqdm(read_text_records(records_path)))

    qdrant_client.upload_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors=vectors,
        payload=payloads,
        ids=None,
        batch_size=BATCH_SIZE,
        parallel=2
    )


if __name__ == '__main__':
    records_path = os.path.join(DATA_DIR, 'docs.jsonl')


