import os
import subprocess
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
import logging

# Load environment variables from a .env file (for securely managing secrets like API keys)
load_dotenv()

# Set OpenAI API key (assuming it's defined in your .env file)
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Set up basic logging configuration with level set to INFO
logging.basicConfig(level=logging.INFO)

def clone_repository(repo_url, local_path):
    """
    (not currently used, pr if you really want to and dm me @didntdrinkwater on tweeter(X)) This function clones a git repository from the provided URL into a local path.
    """
    subprocess.run(["git", "clone", repo_url, local_path])

def is_binary(file_path):
    """
    This function checks whether a file is binary by reading a chunk of the file and looking for null bytes.
    """
    with open(file_path, 'rb') as file:
        chunk = file.read(1024)
    return b'\0' in chunk

def load_docs(root_dir):
    """
    This function walks through a directory, loading text documents and splitting them into chunks.
    """
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            if not is_binary(file_path):
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    doc_chunks = loader.load_and_split()
                    # Prepend the filename to the first chunk
                    if doc_chunks:
                        doc_chunks[0].page_content = f"// {file}\n{doc_chunks[0].page_content}"
                    docs.extend(doc_chunks)
                except Exception as e:
                    logging.error(f"Error loading file {file}: {str(e)}")
    return docs

def split_docs(docs):
    """
    This function splits the loaded documents into smaller chunks of specified size.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1666, chunk_overlap=0)
    return text_splitter.split_documents(docs)

def main(repo_url, root_dir, deep_lake_path):
    """
    This is the main function that loads documents from the specified directory, splits them into chunks, and
    stores them into a DeepLake vector store with embeddings calculated by OpenAI.
    """
    # Print out the directory that the script is going to load documents from
    print(f"Loading documents from directory: {root_dir}")
    
    # Load documents
    docs = load_docs(root_dir)
    print(f"Loaded {len(docs)} documents.")
    
    # Split the documents
    texts = split_docs(docs)
    
    # Initialize embeddings and DeepLake vector store
    embeddings = OpenAIEmbeddings()
    db = DeepLake(dataset_path=deep_lake_path, embedding_function=embeddings)
    
    # Add documents to DeepLake
    db.add_documents(texts)

# Entrypoint of the script
if __name__ == "__main__":
    repo_url = os.environ.get('REPO_URL')
    root_dir = "/mnt/c/Users/hueyfreeman/OneDrive/Desktop/twitter/x/chat-with-repo/the-algorithm" # change me to the repo (clone the repo first)
    deep_lake_path = os.environ.get('DEEPLAKE_DATASET_PATH')
    main(repo_url, root_dir, deep_lake_path)
