from langchain.vectorstores import Pinecone
from langchain.docstore.document import Document
import logging
import pinecone
import time
from typing import Any

from constants import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)


default_pinecone_api_key = os.environ["DEFAULT_PINECONE_API_KEY"]
default_pinecone_environment = os.environ["DEFAULT_PINECONE_ENVIRONMENT"]
default_pinecone_index_name = os.environ["DEFAULT_PINECONE_INDEX_NAME"]


# limit the docs counts to try and avoid the Lambda potentially timing out?!?
max_docs = 1000 
# insert at most this many docs at one time in 'populate_pinecone_index'
# see also: https://docs.pinecone.io/reference/upsert
max_docs_batch_length = 100 


def create_pinecone_index(index_name: str, api_key: str, environment: str, dimension: int, metric='dotproduct', verbose=False) -> None:
    if not environment:
        environment = default_pinecone_environment
    if not index_name:
        index_name = default_pinecone_index_name
        
    pinecone.init(api_key=api_key, environment=environment)
    logger.info("Pinecone initialized")
    if index_name in pinecone.list_indexes():
        logger.info(f"Deleting pre-existing index named '{index_name}'")
        pinecone.delete_index(index_name)
        logger.info(f"Creating index named '{index_name}' with dimension '{dimension}' and metric '{metric}'")
    pinecone.create_index(name=index_name, dimension=dimension, metric=metric)
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)
    if verbose:
        logger.info(f"Created index named '{index_name}' with dimension '{dimension}' and metric '{metric}'")
        index = pinecone.Index(index_name)
        index.describe_index_stats()


def split_list(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
        
def populate_pinecone_index(docs, bedrock_embeddings, index_name, verbose=False):
    """
    populate Pincone index with the document embeddings
    for now: limit this to at most 'max_docs' insertions
    """
    if not index_name:
        index_name = default_pinecone_index_name
    
    vectorstore = retrieve_pinecone_vectorstore(bedrock_embeddings=bedrock_embeddings)
    
    logger.info(f"add docs and their embeddings to pincone index using '{bedrock_embeddings}'/'{index_name}'")
    if len(docs) > max_docs:
        logger.info(f"Truncating docs from {len(docs)} to {max_docs} to avoid potential AWS Lambda timeout")
        docs = docs[:max_docs]
    docs_batches = list(split_list(docs, max_docs_batch_length))
    for idx, docs_batch in enumerate(docs_batches):
        logger.info(f"docs batch length: {len(docs_batch)}")
        logger.info(f"adding docs batch {idx}")
        try:
            vectorstore.add_documents(documents=docs_batch)
        except Exception as e:
            logger.error(f"Failed to add docs batch {idx} into Pinecone index due to {e}")
    
    if verbose:
        try:
            index = pinecone.Index(index_name)
            index.describe_index_stats()
        except:
            logger.error("Failed to describe Pinecone index stats")



def retrieve_pinecone_vectorstore(bedrock_embeddings, index_name=None, api_key=None, environment=None, text_field='text') -> Any:
    """
    register/connect Pinecone index to langchain
    """
    if not api_key:
        api_key = default_pinecone_api_key
    if not environment:
        environment = default_pinecone_environment
    if not index_name:
        index_name = default_pinecone_index_name
        
    pinecone.init(api_key=api_key, environment=environment)
    index = pinecone.Index(index_name)
    vectorstore = Pinecone(index, bedrock_embeddings, text_field)
    return vectorstore


def query_vectorstore(vectorstore, query, k=3) -> Any:
    result = vectorstore.similarity_search(query, k=3) 
    return result
