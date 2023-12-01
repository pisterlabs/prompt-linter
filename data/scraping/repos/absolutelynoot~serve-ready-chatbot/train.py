import os
import chromadb
import shutil
import openai
import sys
import numpy as np
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load the API keys from the .env file
load_dotenv('./.env')
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY') or os.environ.get("LANGCHAIN_API_KEY")
    
# Load the document
print("Loading documents...")
loader = PyPDFDirectoryLoader("./docs")
pages = loader.load()
print(pages[0].page_content)

# Split the document into chuncks
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

splits = text_splitter.split_documents(pages)

# Convert text chucks to embeddings
print("Converting to embeddings...")

embedding = OpenAIEmbeddings(openai_api_key=OPEN_AI_API_KEY)

# Delete vector db if it exists
print("Storing into vector db...")

db_directory = "./docs/vectordb"

if os.path.exists(db_directory):
    shutil.rmtree(db_directory)

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=db_directory
)

# Print the number of vectors stored
print(f"Number of vectors stored: {vectordb._collection.count()}")

# Train success
print("Model Training success!")