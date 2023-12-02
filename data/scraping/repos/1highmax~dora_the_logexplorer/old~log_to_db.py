import re
import hashlib
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# Function to extract log data from a line
def extract_log_data(line):
    pattern = r'(\b\w{3}\s\d{2}\s\d{2}:\d{2}:\d{2}\b)\s+(.*?)\s*:\s*(.*)'
    match = re.match(pattern, line)
    if match:
        return match.groups()
    else:
        return None, None, None

# Function to generate a unique hash for file content
def generate_file_hash(content):
    hasher = hashlib.md5()
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()

def process_log_file(content, persist_directory='db'):


    path = 'streamlit/data/uploaded_log.txt'

    loader = TextLoader('data/uploaded_log.txt', encoding = 'UTF-8')
    doc = loader.load()
    len(doc)
    # OpenAI embeddings
    embedding = OpenAIEmbeddings()

    # Splitting the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(doc)  # Assuming content is a single string

    # Initialize the vector database with the given persist directory
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)
    vectordb.persist()

    print("Vectordb initialized with persist directory:", persist_directory)
    return vectordb
