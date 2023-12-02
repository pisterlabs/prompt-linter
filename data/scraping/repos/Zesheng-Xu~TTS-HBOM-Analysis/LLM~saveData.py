from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain.document_loaders import PyPDFDirectoryLoader


# Load pdf
def loadAndSave():
    directory = ("LLM/data")
     # List all files in the directory
    files = os.listdir(directory)

    loader = PyPDFDirectoryLoader(directory)
    docs = loader.load()

    # Split pdf images into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)

    # Embed pdfs
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vectorstore = FAISS.from_documents(texts, embedding)
    vectorstore.save_local(f"data/TTS_references")
loadAndSave()