import os
import logging
from dotenv import load_dotenv, find_dotenv
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os.path

# Initialize logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def load_environment():
    """Load environment variables."""
    load_dotenv(find_dotenv())
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if openai.api_key is None:
        logging.error("OpenAI API Key not found.")
        exit(1)

def load_documents(directory_path, glob_pattern):
    """Load documents from a directory."""
    try:
        loader = DirectoryLoader(directory_path, glob=glob_pattern)
        return loader.load()
    except Exception as e:
        logging.error(f"Failed to load documents: {e}")
        return []

def split_text_into_chunks(documents, chunk_size=1000, chunk_overlap=0):
    """Split text into smaller chunks."""
    try:
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logging.info("Splitting text into chunk size 1000")
        return text_splitter.split_documents(documents)
    except Exception as e:
        logging.error(f"Failed to split documents: {e}")
        return []

def create_and_persist_embeddings(splits, persist_directory):
    """Create embeddings and persist them."""
    try:
        if os.path.exists(persist_directory):
            logging.info(f"Embeddings already exist in {persist_directory}. Skipping creation.")
            return None

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return vectordb
    except Exception as e:
        logging.error(f"Failed to create and persist embeddings: {e}")
        return None


def create_embeddings():
    # Load environment variables
    load_environment()

    # Load documents
    directory_path = "./src/text_files/"
    glob_pattern = "*.txt"
    documents = load_documents(directory_path, glob_pattern)

    # Split documents into chunks
    splits = split_text_into_chunks(documents)
    
    # Initialize vectordb to None
    vectordb = None

    # Create and persist embeddings only if they don't exist
    persist_directory = './docs/chroma/'
    if not os.path.exists(persist_directory):
        vectordb = create_and_persist_embeddings(splits, persist_directory)

    # Output the count of documents in the collection
    if vectordb:
        logging.info("Successfully created embeddings for all text files")
        logging.info(f"Number of documents in embedding collection: {vectordb._collection.count()}")
    else:
        logging.info("No new embeddings were created.")
        
if __name__ == '__main__':
    create_embeddings()
