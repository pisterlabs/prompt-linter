import os
from typing import List
from urllib.parse import urlparse

import weaviate
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.document_loaders import (
    DirectoryLoader,
    ReadTheDocsLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Chroma, Weaviate

from codinit.config import Secrets, client, secrets
from codinit.utils import download_html

retriever = WeaviateHybridSearchRetriever(
    client, index_name="LangChain_test", text_key="text"
)
# coding_libraries


def compute_embeddings_readthedocs(
    docs_path: str, retriever: WeaviateHybridSearchRetriever
) -> WeaviateHybridSearchRetriever:
    for root, dirs, files in os.walk(docs_path):
        for file in files:
            # os.path.join() will join the directory path and the file name to create a full file path
            filename = os.path.join(root, file)
            print(filename)
            loader = UnstructuredHTMLLoader(filename)
            raw_documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            documents = text_splitter.split_documents(raw_documents)
            print("splitted documents")
            retriever.add_documents(documents)
    return retriever


def get_embedding_store(
    start_urls: List[str] = [
        "https://langchain.readthedocs.io/en/latest/"
    ],  # should be of the form 'https://libname.readthedocs.io..'
    search_index: WeaviateHybridSearchRetriever = retriever,
    secrets: Secrets = secrets,
) -> WeaviateHybridSearchRetriever:
    for start_url in start_urls:
        # The second element in the list should be your library name
        # Remember list index starts from 0
        libname = start_url.split("//")[1].split(".")[0]
        docs_dir = secrets.docs_dir
        docs_path = os.path.join(docs_dir, libname)
        print(f"{libname=}")
        print(docs_path)
        if not os.path.exists(docs_path):
            try:
                os.makedirs(docs_path, exist_ok=True)
                download_html(start_url=start_url, folder=docs_path)
                print("computing embeddings")
                search_index = compute_embeddings_readthedocs(
                    docs_path, retriever=retriever
                )
            except IndexError:
                pass
    return search_index
    """
    # if documentation files exist compute their embeddings
    elif os.path.exists(docs_path):
        search_index = compute_embeddings_readthedocs(docs_path, persist_dir)
    # if documentation files don't exist, download and compute embeddings
    else:
        os.makedirs(docs_path, exist_ok=True)
        download_html(start_url=start_url, folder=docs_path)
        search_index = compute_embeddings_readthedocs(docs_path, persist_dir)
    return search_index"""


def get_read_the_docs_context(query: str, k: int = 5):
    docs = retriever.get_relevant_documents(query=query)
    relevant_docs = ""
    for doc in docs:
        print(doc)
        relevant_docs += doc.page_content
    return relevant_docs
