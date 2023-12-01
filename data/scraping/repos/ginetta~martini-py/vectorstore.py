import os
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.qdrant import Qdrant

qd_client = QdrantClient(url=os.environ.get('QDRANT_URL'), prefer_grpc=False)

logger = logging.getLogger(__name__)

def create_vectorstore_collection(collection_name: str):
    '''
    Create a Qdrant collection with the given name, and the default vector parameters.
    '''
    qd_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=os.environ.get('EMBEDDINGS_DIMENSION_OPENAI'),
            distance=models.Distance.COSINE
        ),
    )

def delete_vectorstore_collection(collection_name: str):
    '''
    Delete a Qdrant collection with the given name.
    '''
    qd_client.delete_collection(collection_name=collection_name)

def get_vectorstore_for_chains(embeddings: OpenAIEmbeddings, collection_name: str) -> Qdrant:
    '''
    Create and return a Langchain-wrapped Qdrant client (vector database),
    optimized for usage in Langchain's chains.
    '''
    return Qdrant(
        client=qd_client,
        collection_name=collection_name,
        embeddings=embeddings,
    )

def delete_points_by_metadata(collection_name: str, doc_name: str, instance_id: int):
    '''
    Delete points (i.e. embeddings) from a Qdrant collection, targeted by metadata "name" and "instance_id".
    '''
    logger.info(f'Deleting embeddings for document "{doc_name}" from collection "{collection_name}"')
    qd_client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key='metadata.instance_id',
                        match=models.MatchValue(
                            value=instance_id
                        )
                    )
                ]
            )
        ),
    )
