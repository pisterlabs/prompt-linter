# Run this script independently (`python create_collection.py`) to create a Qdrant collection. A Qdrant collection is a set of vectors among which you can search.
# All the legal documents over which search needs to be enabled need to be converted to their embedding representation and inserted into a Qdrant collection for search feature to work.

import os
import cohere
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http import models as rest

from constants import (
    ENGLISH_EMBEDDING_MODEL,
    MULTILINGUAL_EMBEDDING_MODEL,
    USE_MULTILINGUAL_EMBEDDING,
    CREATE_QDRANT_COLLECTION_NAME,
)

# load environment variables
QDRANT_HOST = os.environ.get("QDRANT_HOST")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")


def get_embedding_size():
    """
    Get the dimensions of the embeddings returned by the model being used to create embeddings for documents.
        Returns:
            embedding_size (`int`):
                The dimensions of the embeddings returned by the embeddings model.
    """
    if USE_MULTILINGUAL_EMBEDDING:
        embedding_size = 768
    else:
        embedding_size = 4096
    return embedding_size


def create_qdrant_collection(vector_size):
    """
    (Re)-create a Qdrant Collection with the desired `collection name` , `vector_size` and `distance_measure`.
    This collection will be used to keep all the vectors representing all the legal documents.
        Args:
            vector_size (`int`):
                The dimensions of the embeddings that will be added to the collection.
    """
    if USE_MULTILINGUAL_EMBEDDING:
        # multilingual embedding model trained using dot product calculation
        distance_measure = rest.Distance.DOT
    else:
        distance_measure = rest.Distance.COSINE
    print("CREATE_QDRANT_COLLECTION_NAME:", CREATE_QDRANT_COLLECTION_NAME)
    qdrant_client.recreate_collection(
        collection_name=CREATE_QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=distance_measure),
    )


def embed_legal_docs(legal_docs):
    """
    Create embeddings and ids which will used to represent the legal documents upon which search needs to be enabled.
        Args:
            legal_docs (`List`):
                A list of documents for which embeddings need to be created.
        Returns:
            doc_embeddings (`List`):
                A list of embeddings corresponding to each document.
            doc_ids (`List`):
                A list of unique ids which will be used as identifiers for the points (documents) in a qdrant collection.
    """
    if USE_MULTILINGUAL_EMBEDDING:
        model_name = MULTILINGUAL_EMBEDDING_MODEL
    else:
        model_name = ENGLISH_EMBEDDING_MODEL

    legal_docs_embeds = cohere_client.embed(
        texts=legal_docs,
        model=model_name,
    )
    doc_embeddings = [
        list(map(float, vector)) for vector in legal_docs_embeds.embeddings
    ]
    doc_ids = [id for id, _ in enumerate(legal_docs_embeds)]

    return doc_embeddings, doc_ids


def upsert_data_in_collection(vectors, ids, payload):
    """
    Create embeddings and ids which will used to represent the legal documents upon which search needs to be enabled.
        Args:
            vectors (`List`):
                A list of embeddings corresponding to each document which needs to be added to the collection.
            ids (`List`):
                A list of unique ids which will be used as identifiers for the points (documents) in a qdrant collection.
            payload (`List`):
               A list of additional information or metadata corresponding to each document being added to the collection.
    """
    try:
        update_result = qdrant_client.upsert(
            collection_name=CREATE_QDRANT_COLLECTION_NAME,
            points=rest.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payload,
            ),
        )
        return update_result
    except:
        return None


def fetch_legal_documents_and_payload():
    """
    Get the legal documents and additional information (payload) related to them which will be used as part of the search module.
        Returns:
            legal_docs (`List['str]`):
                The documents that will be used as part of the search module.
            payload (`List[Dict]`):
                Additional information related to the documents that are being used as part of the search module.
    """
    legal_dataset = load_dataset("joelito/covid19_emergency_event", split="train")
    legal_docs = legal_dataset["text"]

    # prepare payload (additional information or metadata for documents being inserted)
    payload = list(legal_dataset)

    return payload, legal_docs


if __name__ == "__main__":
    # create qdrant and cohere client
    cohere_client = cohere.Client(COHERE_API_KEY)

    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
    )

    # fetch the size of the embeddings depending on which model is being used to create embeddings for documents
    vector_size = get_embedding_size()

    # create a collection in Qdrant
    create_qdrant_collection(vector_size)

    # load the set of documents which will be inserted into the Qdrant collection
    payload, legal_docs = fetch_legal_documents_and_payload()

    # create embedddings for documents and IDs for documents before insertion into Qdrant collection
    doc_embeddings, doc_ids = embed_legal_docs(legal_docs)

    # insert/update documents in the previously created qdrant collection
    update_result = upsert_data_in_collection(doc_embeddings, doc_ids, payload)

    collection_info = qdrant_client.get_collection(
        collection_name=CREATE_QDRANT_COLLECTION_NAME
    )

    if update_result is not None:
        if collection_info.vectors_count == len(legal_docs):
            print("All documents have been successfully added to Qdrant Collection!")
    else:
        print("Failed to add documents to Qdrant collection")
