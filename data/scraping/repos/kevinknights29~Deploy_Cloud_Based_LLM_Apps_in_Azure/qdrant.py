from __future__ import annotations

import qdrant_client
from langchain.docstore.document import Document
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from src.llm import openai
from src.utils import common

URL = common.config()["vectordb"]["url"]
# if URL == "localhost":
#     URL = common.config()["vectordb"]["localhost_ip"]
PORT = common.config()["vectordb"]["port"]
CLIENT: QdrantClient = None
LANGCHAIN_CLIENT: Qdrant = None

logger = common.create_logger(__name__)


def get_client(url: str = URL, port: int = PORT) -> QdrantClient:
    """
    Get the Qdrant client instance.

    Args:
        url (str): The URL of the Qdrant server. Defaults to the value of URL.
        port (int): The port number of the Qdrant server. Defaults to the value of PORT.

    Returns:
        QdrantClient: The Qdrant client instance.
    """
    global CLIENT
    if CLIENT is None:
        CLIENT = QdrantClient(URL, port=PORT)
    return CLIENT


def get_client_langchain(
    collection: str = None,
    documents: [Document] = None,
) -> Qdrant:
    """
    Get the Qdrant client instance.

    Args:
        connector (str, optional): The connector to use. Defaults to "langchain".
        collection (str, optional): The name of the collection. Required if connector is "langchain".

    Returns:
        QdrantClient: The Qdrant client instance.

    Raises:
        ValueError: If the collection name is not provided when using the "langchain" connector.
    """
    global LANGCHAIN_CLIENT
    if LANGCHAIN_CLIENT is None:
        if collection is None:
            raise ValueError("Collection name must be provided.")
        if documents is None:
            raise ValueError("Documents must be provided.")
        return Qdrant.from_documents(
            documents=documents,
            collection_name=collection,
            embedding=openai.get_embeddings(),
            url=URL,
            force_recreate=True,
        )
    return LANGCHAIN_CLIENT


def _check_if_collection_exist(collection: str) -> bool:
    """
    Check if a collection exists in the database.

    Args:
        collection (str): The name of the collection to check.

    Returns:
        bool: True if the collection exists, False otherwise.
    """
    try:
        response = get_client().get_collection(collection_name=collection)
        if response is not None:
            logger.info("Collection %s exists.", collection)
            return True
    except qdrant_client.http.api_client.UnexpectedResponse:
        logger.info("Collection %s does not exist.", collection)
        return False
    return False


def create_collection(collection: str, documents: list[Document]) -> bool:
    """
    Create a collection in the Qdrant database.

    Args:
        collection (str): The name of the collection to be created.

    Returns:
        bool: True if the collection is created successfully, False otherwise.
    """
    if not _check_if_collection_exist(collection=collection):
        vector_size = len(documents[0].page_content)
        get_client().create_collection(
            collection_name=collection,
            vectors_config={
                "content": rest.VectorParams(
                    distance=rest.Distance.COSINE,
                    size=vector_size,
                ),
            },
        )
        logger.info("Collection %s created.", collection)
        return True
    logger.info("Collection %s already exists.", collection)
    return False


def insert_documents(collection: str, documents: list[Document]) -> None:
    """
    Insert documents into the collection.

    Args:
        collection (str): The name of the collection.
        documents (list): The documents to insert.
    """
    if not _check_if_collection_exist(collection=collection):
        logger.info("Collection %s didnt exist, creating...", collection)
        create_collection(collection=collection, documents=documents)
    try:
        get_client_langchain(collection=collection, documents=documents)
        logger.info("Documents inserted into collection %s.", collection)
    except qdrant_client.http.exceptions.ResponseHandlingException as e:
        raise ValueError("Check your client URL and port.") from e
    except qdrant_client.http.api_client.UnexpectedResponse as e:
        raise ValueError("Check your collection name.") from e


def query_collection(collection: str, documents: [Document], query: str):
    """
    Query the collection.

    Args:
        collection (str): The name of the collection.
        query (str): The query.

    Returns:
        dict: The results of the query.
    """
    response = get_client_langchain(collection, documents).similarity_search(
        query=query,
    )
    return response[0].page_content
