from loguru import logger
from typing import Dict, Iterable
from langchain.vectorstores import FAISS
import time
from langchain.schema import Document

from langchain.embeddings.base import Embeddings
import ai_driver.vector_storage.local_loader as LocalLoader
from ai_driver.config import server_config


def embed_FAISS_from_documents(
    documents: Iterable[Document],
    embedding_model_name: str,
    embedding_model_kwargs: Dict,
) -> FAISS:
    embedder = LocalLoader.get_cache_embeddings(
        embedding_model_name=embedding_model_name,
        embedding_model_kwargs=embedding_model_kwargs,
    )
    vector_store = get_vector_store(embedder, documents)
    return vector_store

def get_vector_store(embedder: Embeddings, documents: Iterable[Document] = None):
    if documents is None:
        documents = LocalLoader.get_default_local_download(server_config.DATA_PATH)
    logger.info("Creating embeddings")
    start_time = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Embeddings created in {elapsed_time:.2f} seconds")

    logger.info("Creating vector store")
    start_time = time.time()
    db_instructEmbedd = FAISS.from_documents(documents, embedder)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Vector store created in {elapsed_time:.2f} seconds")

    logger.info("Vector store created")
    return db_instructEmbedd
