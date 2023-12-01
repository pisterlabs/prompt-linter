# File: db_build.py

# Imports 

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

# Load the data
loader = DirectoryLoader(
    './data/example_pdfs/',
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

# Load document
doc = loader.load()

# Split text from PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
texts = splitter.split_documents(doc)

# Load embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        'device': 'cpu'
    },
)

# Build and persist FAISS vector store
vector_store = FAISS.from_documents(
    texts,
    embeddings,
)
vector_store.save_local('./etc/vector_store/db_faiss')