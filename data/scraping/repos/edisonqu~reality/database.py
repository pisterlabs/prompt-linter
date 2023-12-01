from chromadb.config import Settings
from chromadb.utils.embedding_functions import CohereEmbeddingFunction
import chromadb
import os


def database_initialization_and_collection(cohere_ef):
    collection_name = "test_db"

    chroma_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        # Optional, defaults to .chromadb/ in the current directory
        persist_directory=".chromadb"
    )
    chromadb_client = chromadb.Client(chroma_settings)



    collection = chromadb_client.get_or_create_collection(
        collection_name, embedding_function=cohere_ef)

    return collection