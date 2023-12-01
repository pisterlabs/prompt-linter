# -*- coding: utf-8 -*-
# ! pip install langchain
# ! pip install openai
# ! pip install python-dotenv
# !pip install Pypdf
# ! pip install yt_dlp
# ! pip install pydub
# ! pip install tiktoken

import os
import openai
import sys
import shutil
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("/Users/aaliyahsalia/Desktop/SFBU/6thTrimester/CS589/Week6_HW3/2023Catalog.pdf"),

]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

print(len(splits))

#Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

sentence1 = "i like java"
sentence2 = "i like python"
sentence3 = "the school is nice"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

print(np.dot(embedding1, embedding2))

print(np.dot(embedding1, embedding3))

print(np.dot(embedding2, embedding3))

# # Vectorstores
# ! pip install chromadb

from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/'

# !rm -rf ./docs/chroma  # remove old database files if any

# Remove old database files if any
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)


vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

# Similarity Search
question = "is there an email i can ask for help"

docs = vectordb.similarity_search(question,k=3)

print(len(docs))

print(docs[0].page_content)

vectordb.persist()

# # Failure modes
question = "what is computer engineering?"

docs = vectordb.similarity_search(question,k=5)

print(docs[0])

print(docs[1])

question = "what did they say about CPT/OPT?"

docs = vectordb.similarity_search(question,k=5)

for doc in docs:
    print(doc.metadata)

print(docs[4].page_content)
