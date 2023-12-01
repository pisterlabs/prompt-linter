
import qdrant_client
from django.conf import settings
from langchain.vectorstores import Qdrant

from .embedding import get_embedding


def create_collection(collection_name, texts):
    """
    Create a collection with the given name and texts.

    Parameters:
        collection_name (str): The name of the collection to be created.
        texts (List[str]): A list of texts to be added to the collection.

    Returns:
        None
    """
    embeddings = get_embedding()
    url = settings.QDRANT_URL
    client = qdrant_client.QdrantClient(url=url, prefer_grpc=True)
    client.delete_collection(collection_name)
    Qdrant.from_documents(
        documents=texts,
        embedding=embeddings,
        url=url,
        prefer_grpc=True,
        collection_name=collection_name,
    )
