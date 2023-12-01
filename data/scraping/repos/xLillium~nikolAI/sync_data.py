from qdrant_client import QdrantClient
from qdrant_client.http import models
import streamlit as st
from qdrant_client.http.models import PointStruct
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]
collection_name = st.secrets["QDRANT_COLLECTION_NAME"]

qdrant_client = QdrantClient(
    url=st.secrets["QDRANT_HOST"],
    api_key=st.secrets["QDRANT_API_KEY"],
)

qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=1536, distance=models.Distance.COSINE),
)

with open("data/test.json") as f:
    text = f.read()


def get_chunks(text):
    chunks = []
    while len(text) > 500:
        last_period_index = text[:500].rfind('.')
        if last_period_index == -1:
            last_period_index = 500
        chunks.append(text[:last_period_index])
        text = text[last_period_index+1:]
    chunks.append(text)
    return chunks


chunks = get_chunks(text)

points = []
i = 1
for chunk in chunks:
    i += 1

    response = openai.Embedding.create(
        input=chunk,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    points.append(PointStruct(
        id=i, vector=embeddings, payload={"text": chunk}))

operation_info = qdrant_client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points
)

print(operation_info)
