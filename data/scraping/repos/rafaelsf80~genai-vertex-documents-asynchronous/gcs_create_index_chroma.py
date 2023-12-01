"""
    Creates index in Chroma in Google Cloud Storage, and store it persistently
"""

import logging  
import os
import glob

from google.cloud import storage

import langchain
from langchain.embeddings import VertexAIEmbeddings

print("LangChain version: ",langchain.__version__)

DEST_BUCKET_NAME = "argolis-documentai-unstructured-large-chromadb"
UNIQUE_ID_FOLDER_BLOB = "DOCUMENT_NAME"

REQUESTS_PER_MINUTE = 150

embedding = VertexAIEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)

# Ingest PDF files
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader("output_all.txt") # no funciona si pdf es puro OCR
documents = loader.load()

# split the documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

#docs = text_splitter.create_documents([doc_mexico])
print(f"# of documents = {len(docs)}")

# Store docs in local vectorstore as index
# it may take a while since API is rate limited
from langchain.vectorstores import Chroma

#persist_directory="/Users/rafaelsanchez/git/genai-vertex-unstructured-large-EXTERNAL/local/.chromadb/"
persist_directory=os.path.abspath("./.chromadb")

# Now we can load the persisted database from disk, and use it as normal. 
db = Chroma(collection_name="langchain", persist_directory=persist_directory, embedding_function=embedding)

db.add_documents(documents=docs, embedding=embedding)
db.persist()  

#save chroma db to GCS, keeping folder structure
DIRECTORY_PATH = persist_directory
rel_paths = glob.glob(DIRECTORY_PATH + '/**', recursive=True)

storage_client = storage.Client()
bucket = storage_client.get_bucket(DEST_BUCKET_NAME)
for local_file in rel_paths:
    remote_path = f'{UNIQUE_ID_FOLDER_BLOB}/{"/".join(local_file.split(os.sep)[6:])}'
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)

print(f"Chromadb saved to {DEST_BUCKET_NAME}/{UNIQUE_ID_FOLDER_BLOB}")


# Expose index to the retriever
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k":2})
print(db)
print("DONE")