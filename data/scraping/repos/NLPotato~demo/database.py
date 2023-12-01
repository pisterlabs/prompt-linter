from typing import List

from langchain.schema.vectorstore import VectorStore
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


def create_collection(
    collection_name: str,
    docs: List[Document],
    directory: str = "db/chroma/",
) -> VectorStore:
    """Build a new collection for the local Chroma vectorstore"""
    directory = directory + collection_name
    return Chroma.from_documents(
        collection_name=collection_name,
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=directory,
        collection_metadata={
            "hnsw:space": "cosine"
        },  # default is "l2" (https://docs.trychroma.com/usage-guide)
    )


def create_save_collection(
    collection_name: str,
    docs: List[Document],
):
    import chromadb

    DB_DIR = "db/chroma/"
    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False,
    )
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client_Settings=client_settings,
        persist_direcitory=DB_DIR,
    )
    vectorstore.add_documents(docs, embeddings)
    vectorstore.persist()

    return vectorstore
