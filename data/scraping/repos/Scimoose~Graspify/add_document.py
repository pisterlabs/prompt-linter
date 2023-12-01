# Import necessary modules and libraries
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import sys

# Load environment variables from .env file
load_dotenv()

# Set the value of the OPENAI_API_KEY variable to the value of the environment variable "OPENAI_API_KEY"  # noqa: E501
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

def add_document(document, doc_loader):
    """
     This function loads a document, splits it into smaller chunks, generates embeddings for the text chunks using the OpenAIEmbeddings module and stores them in a Chroma vector store. 
    The vector store is then persisted to disk.
    
    Args:
    - document: a string representing the path to the document to be loaded
    - doc_loader: an instance of a document loader class (e.g. UnstructuredPDFLoader, UnstructuredWordDocumentLoader)

    Returns:
    - None
    """  # noqa: E501
    
    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = 'db'

    # Load the text from a PDF file using the UnstructuredPDFLoader module
    loader = doc_loader(document)
    documents = loader.load()

    # Split the loaded text into smaller chunks using the CharacterTextSplitter module
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings for the text chunks using the OpenAIEmbeddings module and store them in a Chroma vector store  # noqa: E501
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)

    # Persist the vector store to disk
    db.persist()
    db = None

if __name__ == "__main__":
    path = sys.argv[1]
    loaders = {
        ".pdf": UnstructuredPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".md": UnstructuredMarkdownLoader
    }
    ext = os.path.splitext(path)[1]
    if ext in loaders:
        add_document(path, loaders[ext])
        print(f"Done - added {ext} to the database")
    else:
        print("Please provide a .pdf, .docx, or .md file")
