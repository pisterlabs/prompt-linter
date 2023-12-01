""" Chroma DB script to initialize DB with an empty collection"""

import os
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings

if True:
    import sys

    sys.path.append("../../")
from src.utils.config import load_config, set_env_var


def main(debug: bool):
    """Main routine to initialize the AI Assistant database"""
    # loading config file:
    config = load_config(debug=debug)
    # Setting env var for connection
    set_env_var(config)
    # Create the connection
    host = os.getenv("CHROMADB_HOST")
    port = os.getenv("CHROMADB_PORT")
    chroma_client = chromadb.HttpClient(host=host, port=port)
    # test connection
    print("Heartbeat to check connection: ", chroma_client.heartbeat())
    # Creating the collection. If the collection exists it will raise ValueError
    collection_name = config["retriever"]["collection_name"]
    embedding_function = OpenAIEmbeddings()
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"desc": "collection to store ai assistant documents embeddings"},
        embedding_function=embedding_function,
    )
    collection.delete(where={"session_id": "123456"})
    print(collection)


if __name__ == "__main__":
    main(debug=True)
