import weaviate
from weaviate import Config
import weaviate.classes as wvc
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import helper

# Starting up the weaviate client
client = weaviate.Client(
    "http://localhost:8080",
    additional_config=Config(grpc_port_experimental=50051)
)

print(client.collection.delete("Podcast"))

# Creating a new collection named Podcast
client.collection.create(
    name="Podcast",
    properties=[
        wvc.Property(
            name="title",
            data_type=wvc.DataType.TEXT,
        ),
        wvc.Property(
            name="transcript",
            data_type=wvc.DataType.TEXT,
        )
    ],
    vectorizer_config=wvc.ConfigFactory.Vectorizer.text2vec_transformers()
)

# Checking if the collection is created successfully
print(client.collection.exists("Podcast"))

with open("./data/podcast_ds.json", 'r', encoding = 'utf-8') as f:
    datastore = json.load(f)

podcast = client.collection.get("Podcast")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)

transcripts_to_add = []

#Chunking the data as the transcripts are too long and won't give good performance for semantic search
with helper.std_out_err_redirect_tqdm() as orig_stdout:
    for data in tqdm(datastore, desc='Importing transcripts', file = orig_stdout, unit = 'transcript'):
        transcripts_to_add = []
        chunked_transcript = text_splitter.create_documents([data["transcript"]])
        for chunk in chunked_transcript:
            transcripts_to_add.append(
                wvc.DataObject(
                    properties={
                        "title": data["title"],
                        "transcript": chunk.page_content,
                    }
                )
            )
        response = podcast.data.insert_many(transcripts_to_add)
        message = str(data["title"]) + ' imported'
        helper.log(message)
        print(response.errors)

print(podcast.query.fetch_objects(limit=2))