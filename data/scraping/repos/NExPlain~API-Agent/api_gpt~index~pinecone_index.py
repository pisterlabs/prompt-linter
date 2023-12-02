import os
from timeit import default_timer as timer
from typing import List

import pinecone
from dotenv import load_dotenv
from flask import current_app
from langchain.llms.openai import OpenAIChat
from llama_index import LLMPredictor, ServiceContext, VectorStoreIndex, download_loader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.schema import NodeWithScore
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
from sqlalchemy import true

from api_gpt.services.constants import CHATGPT_MODEL

# Load environment variables from .env file
load_dotenv()

if os.environ["PINECONE_API_KEY"] is None:
    raise ValueError("OPENAIPINECONE_API_KEY_API_KEY environment variable not set")
if os.environ["PINECONE_ENVIRONMENT"] is None:
    raise ValueError("PINECONE_ENVIRONMENT environment variable not set")
if os.environ["PINECONE_INDEX_NAME"] is None:
    raise ValueError("PINECONE_INDEX_NAME environment variable not set")

API_KEY = os.environ["PINECONE_API_KEY"]
ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)
try:
    pinecone.create_index(INDEX_NAME, dimension=1536)
except Exception as e:
    print(f"Error when try to create pinecone index {INDEX_NAME}: {e}")
PINECONE_INDEX = pinecone.Index(INDEX_NAME)
PINECONE_EMBEDDING = OpenAIEmbedding()

llm_predictor_chatgpt = LLMPredictor(
    llm=OpenAIChat(temperature=0, model_name=CHATGPT_MODEL),
)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor_chatgpt,
    embed_model=PINECONE_EMBEDDING,
    chunk_size=1536,
    chunk_overlap=20,
)


def get_email_documents(user_id: str, email_query: str):
    GmailReader = download_loader("GmailReader")
    loader = GmailReader(query=email_query)
    documents = loader.load_data()
    for document in documents:
        document.metadata = {"user_id": user_id}
    current_app.logger.debug(f"Loaded {len(documents)} documents")
    return documents


def upsert_documents(user_id, documents) -> VectorStoreIndex:
    current_app.logger.debug(f"Start creating pinecone index")
    start = timer()
    metadata_filters = {"user_id": user_id}
    vector_store = PineconeVectorStore(
        pinecone_index=PINECONE_INDEX,
        metadata_filters=metadata_filters,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # current_app.logger.debug(f"documents : {documents}")

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )
    end = timer()
    seconds = int(end - start)
    current_app.logger.debug(f"finished creating index, took {seconds} seconds")
    return index


def get_pinecone_index(user_id: str | None) -> VectorStoreIndex:
    if user_id is None:
        vector_store = PineconeVectorStore(
            PINECONE_INDEX,
        )
    else:
        metadata_filters = {"user_id": user_id}
        vector_store = PineconeVectorStore(
            PINECONE_INDEX,
            metadata_filters=metadata_filters,
        )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def query(user_id: str | None, query: str) -> str:
    index = get_pinecone_index(user_id)
    query_engine = index.as_query_engine()
    return str(query_engine.query(query))


def retrieve(user_id: str | None, query: str) -> List[NodeWithScore]:
    if user_id is None:
        index = get_pinecone_index(user_id)
        retriever = index.as_retriever()
    else:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="user_id", value=user_id)]
        )

        index = get_pinecone_index(user_id)
        retriever = index.as_retriever(filters=filters)
    return retriever.retrieve(query)


def delete(user_id: str | None):
    if user_id is None:
        PINECONE_INDEX.delete(delete_all=True)
    else:
        return PINECONE_INDEX.delete(filter={"user_id": user_id})
