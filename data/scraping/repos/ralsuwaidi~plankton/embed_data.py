from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from typing import List
from langchain.docstore.document import Document
import shutil


DATABASE_DIR = "chroma_db"
DB_COLLECTION = "plankton_1"


# Define the base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the path to the .env file
dotenv_path = os.path.join(BASE_DIR, ".env")

# Load the .env file
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_embeddings(
    max_retries=100, request_timeout=20000, show_progress_bar=True
) -> OpenAIEmbeddings:
    # Retrieve the API key from environment variable
    model_name = "text-embedding-ada-002"

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY,
        max_retries=max_retries,  # large retries to deal with rate
        request_timeout=request_timeout,
        show_progress_bar=show_progress_bar,
    )
    return embed


def get_vector_store(
    embedding_function: OpenAIEmbeddings,
    persist_directory=DATABASE_DIR,
    collection_name=DB_COLLECTION,
):
    """
    This function creates and returns a vector store (Chroma) instance
    using the provided embedding function, persist directory, and collection name.
    """
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )


def embed_data(
    embedding: OpenAIEmbeddings,
    docs: List[Document] = None,
    persist_directory=DATABASE_DIR,
    collection_name=DB_COLLECTION,
    delete_existing_db=False,
) -> Chroma:
    """
    This function either creates a new Chroma instance by embedding the
    supplied documents or retrieves an existing vector store from the specified location.
    If delete_existing_db is True and the DATABASE_DIR already exists,
    the existing directory and its contents are deleted before embedding documents.
    """
    # Delete the existing DB if requested
    if delete_existing_db and os.path.exists(DATABASE_DIR):
        shutil.rmtree(DATABASE_DIR)

    # If the persisting directory doesn't exist, create a new Chroma from the documents.
    # Otherwise, get the Chroma instance from the existing vector store.
    return (
        Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        if not os.path.exists(DATABASE_DIR)
        else get_vector_store(
            embedding_function=embedding,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    )
