from qdrant_client import QdrantClient
from qdrant_client.grpc.qdrant_pb2_grpc import Qdrant
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import openai
import os

#from dotenv import load_dotenv

import ix as ix

#load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")



openai.api_key = "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"
#openai.api_key = '{OPEN_API_KEY}'

qdrant_client = QdrantClient(
    url="https://9c258883-7f0b-40bc-9830-f81a215c19f2.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="nkjC0SgmVYjT1vhHl8HZ3BHdnOQVC4IgWunB3MELFJSIbYYpfhgADA",
)

qdrant_client.recreate_collection(
    collection_name="iching",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
)

#print("Create collection reponse:", qdrant_client)

collection_info = qdrant_client.get_collection(collection_name="iching")

#print("Collection info:", collection_info)

#
import pdfplumber

fulltext = ""
with pdfplumber.open("ix2.pdf") as pdf:
    # loop over all the pages
    for page in pdf.pages:
        fulltext += page.extract_text()

#print(fulltext)

#
text = fulltext

chunks = []
while len(text) > 500:
    last_period_index = text[:500].rfind('.')
    if last_period_index == -1:
        last_period_index = 500
    chunks.append(text[:last_period_index])
    text = text[last_period_index+1:]
chunks.append(text)


for chunk in chunks:
    print(chunk)
    #print("---")

#

from qdrant_client.http.models import PointStruct

points = []
i = 1
for chunk in chunks:
    i += 1

    #print("Embeddings chunk:", chunk)
    response = openai.Embedding.create(
        input=chunk,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    points.append(PointStruct(id=i, vector=embeddings, payload={"text": chunk}))

#

operation_info = qdrant_client.upsert(
    collection_name="iching",
    wait=True,
    points=points
)

#print("Operation info:", operation_info)

#

def create_answer_with_context(query):
    response = openai.Embedding.create(
        input="how can I know myself using the i-ching?",
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    search_result = qdrant_client.search(
        collection_name="iching",
        query_vector=embeddings,
        limit=5
    )

    prompt = "Context:\n"
    for result in search_result:
        prompt += result.payload['text'] + "\n---\n"
    prompt += "Question:" + query + "\n---\n" + "Answer:"

    #print("----PROMPT START----")
    #print(":", prompt)
    #print("----PROMPT END----")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
        )

    return completion.choices[0].message.content


input = ix.sign
answer = create_answer_with_context("if I have the 48 genekeys in my lifes work what that means?")
print(answer)

#q = Qdrant.load_model('path_to_qdrant_model')