from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Milvus
import streamlit as st
from pymilvus import connections, utility


@st.cache_resource
def load_collections() -> dict[str, Milvus]:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings.encode_kwargs = dict(normalize_embeddings=True)
    connection_args = {"host": "localhost", "port": "19530"}
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    collections = ["alz", "pdf"]
    vector_stores = dict()
    connections.connect(**connection_args)
    for collection in collections:
        if utility.has_collection(collection):
            vector_store = Milvus(
                embedding_function=embeddings,
                collection_name=collection,
                connection_args=connection_args,
                search_params=search_params,
            )
            vector_stores[collection] = vector_store
    return vector_stores
