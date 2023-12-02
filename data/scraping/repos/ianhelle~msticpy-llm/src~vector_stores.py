# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Vector stores module."""
import logging
from pathlib import Path
from typing import Optional

from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from _version import VERSION

__version__ = VERSION
__author__ = "Ian Hellen"

# TODO: move to config
MP_RTD_PATH = "E:/src/msticpy/docs/build/html/"
MP_VS_PATH = "./mp-docs-vs.faiss_index"

logger = logging.getLogger("msticpy-docs-loader")


def read_vector_store(vs_path: str, caller: str = "unknown") -> VectorStore:
    """Read pickled vectorstore from file."""
    logger.info("loading vectorstore")
    if not Path(vs_path).exists():
        raise ValueError(f"{vs_path} does not exist, please run '{caller}' first")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(vs_path, embeddings)
    logger.info("vectorstore %s loaded", vs_path)
    return vectorstore


def create_vector_store(loader, vs_path: Optional[str] = None):
    """Create vector store from data."""
    logger.info("Loading documents")
    data = loader.load()
    logger.info("Splitting documents with RecursiveCharacterTextSplitter")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(data)
    logger.info("Creating vectorstore")
    embeddings = OpenAIEmbeddings()
    print("Creating vectorstore. This may take a while...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    if vs_path:
        # Save vectorstore
        vectorstore.save_local(vs_path)
        logger.info("Saved vectorstore to %s", vs_path)
    return vectorstore
