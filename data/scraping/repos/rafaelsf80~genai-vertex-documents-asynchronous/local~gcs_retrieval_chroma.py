"""
    Retrieval process using langChain, Chroma and Vertex AI LLM (text-bison@001)
    Chroma Db is downloaded from GCS
"""

import os
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings

from google.cloud import storage

from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings


BUCKET_NAME = "argolis-documentai-unstructured-large-chromadb"
PREFIX = 'DOCUMENT_NAME/'

REQUESTS_PER_MINUTE = 150

llm = VertexAI(
    model_name='text-bison@001',
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,top_k=40,
    verbose=True,
)

embedding = VertexAIEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)

from langchain.vectorstores import Chroma

# Init Chromadb
# db = chromadb.Client(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="/Users/rafaelsanchez/git/genai-vertex-unstructured-EXTERNAL/.chromadb/" # Using full path for debugging
# ))

# download gcs folder, keeping folder structure
storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET_NAME)
blobs = bucket.list_blobs(prefix=PREFIX)  # Get list of files
for blob in blobs:
    if blob.name.endswith("/"):
        continue
    file_split = blob.name.split("/")
    directory = "/".join(file_split[0:-1])
    Path(directory).mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(blob.name) 

persist_directory=os.path.abspath(f"./{PREFIX}.chromadb")
# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(collection_name="langchain", persist_directory=persist_directory, embedding_function=embedding)


# Expose index to the retriever
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k":2})

# Create chain to answer questions
from langchain.chains import RetrievalQA
from langchain import PromptTemplate


# Uses LLM to synthesize results from the search index.
# We use Vertex PaLM Text API for LLM
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    # chain_type_kwargs={"prompt": PromptTemplate(
    #         template=template,
    #         input_variables=["context", "question"],
    #     ),},
    return_source_documents=True)

query = "What was BBVA net income in 2022?"

result = qa({"query": query})
print(result)

print(qa.combine_documents_chain.llm_chain.prompt.template)