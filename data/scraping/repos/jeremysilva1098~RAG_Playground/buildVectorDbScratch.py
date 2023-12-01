'''Build the vectordb from scratch'''
import pandas as pd
import openai
import json
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os
import requests

load_dotenv("../.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# create the vectordb client
dbClient = QdrantClient(host='localhost', port=6333)
collection = 'wiki_movies'
localUrl = "http://localhost:6333"
# create the collection
dbClient.create_collection(collection_name=collection,
                           vectors_config=models.VectorParams(
                               size=1536, distance=models.Distance.COSINE)
                        )


def get_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002",
        )
        return response["data"][0]["embedding"]
    except:
        print("Error with text: ", text)
        return None


df = pd.read_csv("data/wiki_movie_plots_deduped.csv")
print(f"Number of records: {len(df)}")
#print(df.columns)
# convert to a dict
data_dict = df.to_dict(orient='records')
#print(json.dumps(data_dict[:10], indent=4))

documents = []

for rec in data_dict:
    payload = {
        "release_year": rec["Release Year"],
        "title": rec["Title"],
        "origin": rec["Origin/Ethnicity"],
        "genere": rec["Genre"],
        "url": rec["Wiki Page"]
    }
    director = rec["Director"].split(",")
    director = [d.strip() for d in director]
    payload["director"] = director
    # add the cast
    if isinstance(rec["Cast"], float):
        cast = []
    else:
        cast = rec["Cast"].split(",")
        cast = [c.strip() for c in cast]
    payload["cast"] = cast
    # add the text
    text = rec["Plot"]
    payload["text"] = text
    documents.append(payload)


# embed and upload
dbClient.upload_records(
    collection_name=collection,
    records=[
        models.Record(
            id=idx,
            vector=get_embedding(doc['text']),
            payload=doc
        ) for idx, doc in enumerate(documents)
    ]
)


# snapshot the db and write to disk
res = requests.post(localUrl + "/snapshots")
print(res.json())
print(res.text)

snapshots = dbClient.list_full_snapshots()
print(snapshots)

# download the latest snapshot
res = requests.get(localUrl + "/snapshots/" + snapshots[0].name)
# write snapshot to disk
with open("data/" + "latest.snapshot", "wb") as f:
    f.write(res.content)


