import json

from langchain.document_loaders import JSONLoader  # pip install jq
from langchain.embeddings.openai import OpenAIEmbeddings  # pip install tiktoken
from qdrant_client import QdrantClient  # pip install qdrant-client
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema.document import Document
from uuid import uuid4

import os
from aidevs_other_stuff.jsonReader import JsonReader

class QdrantHelper:
    def init_collection(persistent_memory, collection_name, limit=300):

        qdrant = QdrantClient(
            "http://localhost:6333")  # url=os.environ['QDRANT_URL']) |qdrant = QdrantClient("localhost", port=6333)
        result = qdrant.get_collections()
        indexed = next((collection for collection in result.collections if collection.name == collection_name), None)
        print("Collections:", result)
        if not indexed:
            qdrant.create_collection(collection_name,
                                     vectors_config=VectorParams(size=1536, distance="Cosine", on_disk=True))
        collection_info = qdrant.get_collection(collection_name)
        if not collection_info.points_count:

            print("read", len(persistent_memory), "documents")
            documents = [Document(page_content=str(content),
                                  metadata={'source': collection_name, 'uuid': str(uuid4()), 'content': content}) for
                         content in persistent_memory]

            points = []
            embeddings = OpenAIEmbeddings()
            for document in documents:
                embedding = embeddings.embed_documents([document.page_content])[0]
                points.append({
                    'id': document.metadata['uuid'],
                    'payload': document.metadata,
                    'vector': embedding,
                })
            print("Created", len(points), "vectors")

            # upload portions of 300
            nr = 1
            batch_size = 300
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                qdrant.upsert(collection_name, wait=True, points=batch)
                print("Indexed", batch*nr, "documents")
                nr+=1


