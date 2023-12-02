import json

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings  # pip install tiktoken
from qdrant_client import QdrantClient  # pip install qdrant-client
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema.document import Document
from uuid import uuid4

import os

# https://qdrant.tech/documentation/quick-start/
# This code initializes Qdrant collection ai_devs, and adds vectors from local documents to it
# then it searches the collection for the most similar vector to the query vector
# vectors for documents and the query are generated using OpenAI API ada / creating embeddings
if __name__ == '__main__':
    MEMORY_PATH = "init_data.md"
    COLLECTION_NAME = "ai_devs"

    # create embeddings for sample text query
    embeddings = OpenAIEmbeddings()
    query = "Do you know the name of Adam's dog?"
    queryEmbedding = embeddings.embed_query(query)

    qdrant = QdrantClient("http://localhost:6333")  # url=os.environ['QDRANT_URL']) |qdrant = QdrantClient("localhost", port=6333)
    result = qdrant.get_collections()
    indexed = next((collection for collection in result.collections if collection.name == COLLECTION_NAME), None)
    print("Collections:", result)

    if not indexed:
        qdrant.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=1536, distance="Cosine", on_disk=True))

    collectionInfo = qdrant.get_collection(COLLECTION_NAME)

    if not collectionInfo.points_count:
        loader = TextLoader(MEMORY_PATH)
        memory = loader.load()[0]
        documents = [Document(page_content=content,metadata={'source':COLLECTION_NAME, 'uuid':str(uuid4()), 'content': content}) for content in memory.page_content.split("\n\n")]

        points = []
        for document in documents:
            embedding = embeddings.embed_documents([document.page_content])[0]
            points.append({
                'id': document.metadata['uuid'],
                'payload': document.metadata,
                'vector': embedding,
            })

        qdrant.upsert(COLLECTION_NAME, wait=True, points=points)

    search = qdrant.search(COLLECTION_NAME, query_vector=queryEmbedding, limit=1, query_filter={
        'must': [
            {
                'key': 'source',
                'match': {
                    'value': COLLECTION_NAME
                }
            }
        ]
    })

    print(search)
