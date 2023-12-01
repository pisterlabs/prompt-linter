import logging
from langchain.document_loaders import DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from config import (
    PERSIST_DIRECTORY,
    MODEL_DIRECTORY,
    SOURCE_DIR,
    EMBEDDING_MODEL,
    DEVICE_TYPE,
    CHROMA_SETTINGS,
)

def load_docs(directory: str = SOURCE_DIR):
    """
    Loads documents from a specified directory.
    
    Args:
        directory (str): The directory path containing PDF documents.
    
    Returns:
        list: A list of loaded documents.
    """
    loader = DirectoryLoader(directory, glob="**/*.pdf", use_multithreading=True, loader_cls=PDFMinerLoader)
    docs = loader.load()
    logging.info(f"Loaded {len(docs)} documents from {directory}")
    return docs

def split_docs(documents,chunk_size=1000,chunk_overlap=200):
    """
    Splits documents into smaller chunks for processing.
    
    Args:
        documents (list): List of documents to be split.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between adjacent chunks.
    
    Returns:
        list: List of split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    logging.info(f"Split {len(docs)} documents into chunks")
    return docs

def builder():
    """
    Builds the database by loading, splitting, and embedding documents.
    """
    logging.info("Building the database")
    documents = load_docs()
    docs = split_docs(documents)

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE_TYPE},
        cache_folder=MODEL_DIRECTORY,
    )
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,

    )
    logging.info(f"Loaded Documents to Chroma DB Successfully")


if __name__ == '__main__':
 
    builder()