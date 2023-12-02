import sys
import os
import chromadb
from chromadb.config import Settings
from pprint import pprint as pp
from dotenv import load_dotenv
from openai_embedding_function import get_openai_ef

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY")


def pretty_query_results(results) -> dict:
    new_results = list()
    for i in range(0, len(results)):
        new_result = dict()
        new_result["distance"] = results["distances"][0][i]
        new_result["metadata"] = results["metadatas"][0][i]
        new_result["metadata"] = results["metadatas"][0][i]
        new_result["id"] = results["ids"][0][i]
        new_results.append(new_result)
    return new_results


def find(collection: chromadb.Client,
         search_term: str):

    res = collection.query(
        query_texts=search_term,
        n_results=5)
    pp(pretty_query_results(res))


def find_formatted(collection: chromadb.Client,
                   search_term: str, n_results: int = 5):

    res = collection.query(
        query_texts=search_term,
        n_results=n_results)
    return pretty_query_results(res)


if __name__ == "__main__":
    load_dotenv()
    openai_ef = get_openai_ef(os.getenv("OPENAI_API_KEY"))
    chroma_client = chromadb.Client(
        Settings(persist_directory=PERSIST_DIRECTORY, chroma_db_impl="duckdb+parquet"))
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME)

    search_term = sys.argv[1]
    find(collection, search_term)
