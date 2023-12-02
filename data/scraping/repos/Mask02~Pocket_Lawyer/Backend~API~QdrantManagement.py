import json
import os
import uuid
from typing import Dict, List
import openai
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
import Creds


class QdrantManagement:

    def __init__(
            self,
            qdrant_api_key: str = Creds.QDRANT_API_KEY,
            qdrant_host: str = Creds.QDRANT_HOST,
            cohere_api_key: str = Creds.COHERE_API_KEY,
            openai_api_key: str = Creds.OPENAI_API_KEY,
            collection_name: str = "Pocket_lawyer",
    ):
        self.qdrant_client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.co_client = cohere.Client(api_key=cohere_api_key)
        openai.api_key = openai_api_key

    # create/recreate collection on Qdrant
    def createCollection(self, collection_name):
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=Creds.COHERE_SIZE_VECTOR, distance=models.Distance.COSINE
            ),
        )

    # Get collection info from Qdrant
    def getCollectionInfo(self, collection_name):
        return self.qdrant_client.get_collection(collection_name)

    # Qdrant requires data in float format
    def _float_vector(self, vector: List[float]):
        return list(map(float, vector))

    # Embedding using Cohere Embed model
    def _embed(self, text):
        return self.co_client.embed(texts=text, model="large", truncate="RIGHT").embeddings

    # Prepare Qdrant Points
    def _qdrant_format(self, data):
        # For cohere embeddings, data should list of strings
        embeds = self._embed(data)

        points = [
            models.PointStruct(
                id=uuid.uuid4().hex,
                payload={"text": point},
                # Can add additional details about text here like article no, chapter no, title etc.
                vector=self._float_vector(embeds[index]),
            )
            for index, point in enumerate(data)
        ]

        return points

    # Upload dataset embeddings in the form of points to Qdrant 'legalcompanion' Collection
    def uploadData(self, data):
        points = self._qdrant_format(data)

        result = self.qdrant_client.upsert(
            collection_name=self.collection_name, points=points
        )

        return result


if __name__ == "__main__":
    qdrant = QdrantManagement()

    qdrant.createCollection("Pocket_lawyer")
    dataset = ""
    with open("Dataset.json", 'r') as f:
        dataset = f.read()
    data = json.loads(dataset)["data"]

    data_list = list()
    for description in data:
        data_list.append(description["Description"])

    qdrant.uploadData(data_list)

    # print(QdrantManagement.getCollectionInfo('legalcompanion'))
