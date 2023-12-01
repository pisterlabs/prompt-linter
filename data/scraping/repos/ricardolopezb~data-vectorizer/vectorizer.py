import os
import uuid
import pinecone
from openai import OpenAI
import pandas as pd
import requests

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENVIRONMENT"))

df = pd.read_csv(os.environ.get("FILENAME"), delimiter=';')

rows_as_strings = df.apply(lambda row: ';'.join(map(str, row)), axis=1).tolist()

vectors = []

for row in rows_as_strings:
    row_fields = row.split(';')
    vectors.append({
        "id": str(uuid.uuid4()),
        "values": client.embeddings.create(input=row, model="text-embedding-ada-002").data[0].embedding,
        "metadata": {
            "Id": row_fields[0],
            "Symptom": row_fields[1],
            "Description": row_fields[2]
        }
    })

url = os.environ.get("PINECONE_CONNECTION_URL") + "/vectors/upsert"

payload = {"vectors": vectors}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Api-Key": os.environ.get("PINECONE_API_KEY")
}


index = pinecone.Index(os.environ.get("PINECONE_INDEX_NAME"))
print("INDEX", os.environ.get("PINECONE_INDEX_NAME"))
print("Upserting")
response = requests.post(url, json=payload, headers=headers)
print(response.text)
