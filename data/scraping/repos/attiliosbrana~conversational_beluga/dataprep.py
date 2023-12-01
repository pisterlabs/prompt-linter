#!/usr/bin/env python3
# coding: utf-8
import glob
import os

from dotenv import load_dotenv
from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# Get Env Variables


load_dotenv()  # load the values for environment variables from the .env file

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")

# Step1: Define a sentence transformer model that will be used
#        to convert the documents into vector embeddings
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

# Step2: Create a list of html contents from the documents
html_docs = []
path_to_dir = f"./docs/"
html_files = glob.glob(os.path.join(path_to_dir, "*"))
for _file in html_files:
    with open(_file) as f:
        loader = BSHTMLLoader(_file)
        data = loader.load()
        html_docs.extend(data)

# Step3: Create & save a vector database with the vector embeddings
#        of the documents
db = FAISS.from_documents(html_docs, embeddings)
db.save_local("faiss_index")
