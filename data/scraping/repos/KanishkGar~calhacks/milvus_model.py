import os
import random
from typing import List, Dict, Union
import openai

PICKLE_PATH = os.path.join(os.path.dirname(__file__), "../data/")

# Milvus contents
COLLECTION_NAME = 'title_db'  # Collection name
DIMENSION = 1536  # Embeddings size
COUNT = 100  # How many titles to embed and insert.
MILVUS_HOST = 'localhost'  # Milvus server URI
MILVUS_PORT = '19530'
OPENAI_ENGINE = 'text-embedding-ada-002'  # Which engine to use
 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

print(OPENAI_API_KEY)

# connect to milvus
from milvus import default_server
from pymilvus import (
    connections, utility, 
    MilvusClient, 
    FieldSchema, DataType, CollectionSchema, Collection
)

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# if utility.has_collection(COLLECTION_NAME):
#     utility.drop_collection(COLLECTION_NAME)

fields = [
    # id format: class:lecture:start_second:end_second - eg. cs162:1:543:600
    FieldSchema(name='id', dtype=DataType.VARCHAR, description='Ids', is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name='caption', dtype=DataType.VARCHAR, description='Title texts', max_length=2000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=DIMENSION)
]
schema = CollectionSchema(fields=fields, description='Title collection')
collection = Collection(name=COLLECTION_NAME, schema=schema)

index_params = {
    'index_type': 'IVF_FLAT',
    'metric_type': 'L2',
    'params': {'nlist': 1024}
}
collection.create_index(field_name="embedding", index_params=index_params)

collection.load()

def embed(text):
    return openai.Embedding.create(
        input=text, 
        engine=OPENAI_ENGINE)["data"][0]["embedding"]

def get_completion(summarize_caption: str):
    return summarize_caption
    prev_messages: List[Dict[str, str]] = [{"role": "system", "content": "You summarize the lecture text that is passeed to you as if you were the professor"}]
    new_message = [{"role": "user", "content": summarize_caption}]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prev_messages + new_message
    )
    return completion.choices[0].message.content

class MilvusModel:
    @classmethod
    def search(cls, text):
    # Search parameters for the index
        search_params={
            "metric_type": "L2"
        }

        results=collection.search(
            data=[embed(text)],  # Embeded search value
            anns_field="embedding",  # Search across embeddings
            param=search_params,
            limit=5,  # Limit to five results per search
            output_fields=['caption']  # Include caption field in result
        )

        ret=[]
        for hit in results[0]:
            row = [hit.id, hit.score, get_completion(hit.entity.get('caption'))]
            ret.append(row)
        return ret
    