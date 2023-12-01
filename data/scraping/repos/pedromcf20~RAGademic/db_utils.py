import logging

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def initialize_chroma(splits: list[str], embeddings: OpenAIEmbeddings, persist_directory: str = "db/chroma") -> Chroma:
    """
    Initializes a Chroma vector store from document splits.

    Args:
        splits (list[str]): List of split document chunks.
        embeddings (OpenAIEmbeddings): Embeddings object.
        persist_directory (str): Directory to persist the vector store.

    Returns:
        Chroma: Initialized Chroma vector store.
    """
    try:
        # Initialize Chroma vector store with document splits and persist it
        vectordb = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
        vectordb.persist()
        return vectordb
    except Exception as e:
        # Log any exceptions during Chroma initialization
        logging.error(f'Error initializing Chroma: {e}')
        return None
