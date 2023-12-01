import os
import scrapy
from scrapy.crawler import CrawlerProcess
from qdrant_client import models, QdrantClient
import cohere
from qdrant_client.http import models as rest
import pandas as pd
import json
import xml.etree.ElementTree as ET
import requests
import time
import openai

os.environ["COHERE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.getenv("OPENAI_API_KEY")

# Qdrant
qdrant = QdrantClient(
    url="", 
    prefer_grpc=True,
    api_key="",
)


def embed_text(text: list, model='text-embedding-ada-002'):
    """Generate text embeddings using OpenAI's Ada model."""
    if type(text) is str:
        text = [text]
    # Since you are passing a list of texts, you might need to loop through each text item in the list
    vectors = []
    for t in text:
        response = openai.Embedding.create(input=t, model=model)
        embedding = response['data'][0]['embedding']
        vectors.append(list(map(float, embedding)))
    print("Embedded.")
    return vectors

def create_collection(name):
    # Create Qdrant vector database collection
    qdrant.recreate_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=1536, 
            distance=rest.Distance.COSINE
        ),
    )
    print(f"The collection {name} was created!")

def load_from_scrapy_output():
    print("\n\n\nStarted loading from the json file.\n")
    with open("output.json", "r") as f:
        chunks = [item["text"] for item in json.load(f)]

    data = []
    for idx, chunk in enumerate(chunks):
        data.append({
            'id': idx + 1,
            'content': chunk,
            'version': 'v1'
        })

    df = pd.DataFrame(data)
    ids = df["id"].tolist()
    vectors = embed_text(df["content"].tolist())

    qdrant.upsert(
    collection_name="colby", 
    points=rest.Batch(
        ids=ids,
        vectors=vectors,
        payloads=df.to_dict(orient='records'),
    ))

if __name__ == "__main__":
    create_collection("colby")
    print("Created a collection")

    # Load the scraped data to Qdrant
    load_from_scrapy_output()
    print("Successfully uploaded to QDrant")
