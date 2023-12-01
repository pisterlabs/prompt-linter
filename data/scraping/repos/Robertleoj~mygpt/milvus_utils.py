from typing import List
from langchain.schema.document import Document
from config import get_config
from pymilvus import (
    utility,
    Collection,
    connections,
    FieldSchema,
    DataType,
    CollectionSchema
)
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings

def connect():
    alias = get_config("MILVUS_ALIAS")
    host = get_config("MILVUS_HOST")
    port = get_config("MILVUS_PORT")

    connections.connect(
        alias=alias, 
        host=host, 
        port=port
    )

def make_collection(name: str):

     openai_emb_dim = get_config("OPENAI_EMBEDDING_DIM", 1536)

     fields = [
         FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
         FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=200),
         FieldSchema(name="file", dtype=DataType.VARCHAR, max_length=200),
         FieldSchema(name="last_modified", dtype=DataType.INT64),
         FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
         FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=openai_emb_dim)
     ]
     schema = CollectionSchema(fields, "Stores documents")
     documents = Collection(name=name, schema=schema)

     index_params = {
         "index_type": "IVF_FLAT",
         "metric_type": "L2",
         "params": {"nlist": 128},
     }   

     documents.create_index(field_name="vector", index_params=index_params)

def get_collection():

    collection_name = get_config("MILVUS_COLLECTION_NAME", "test")

    if not utility.has_collection(collection_name):
        make_collection(collection_name)


    return Collection(name=collection_name)
        

def get_vectorstore():

    collection_name = get_config("MILVUS_COLLECTION_NAME", "test")

    connect()
    get_collection()

    emb_model = OpenAIEmbeddings()

    vector_store = Milvus(
        embedding_function=emb_model,
        collection_name=collection_name,
        connection_args={
            "host": get_config("MILVUS_HOST"),
            "port": get_config("MILVUS_PORT"),
        }
    )

    return vector_store

       
        

