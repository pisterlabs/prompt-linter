import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/Assura-Basis_CGA_LAMal_2024_F.pdf"),
    PyPDFLoader("docs/CGA_F.pdf"),
    PyPDFLoader("docs/Vue_Ensemble_Produits_F_V33_08.2023.pdf"),
    PyPDFLoader("docs/NEW_ASSURA_CSC-Natura_2015.07_F.pdf")
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

# Embeddings

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

print(f'Sentence 1: "{sentence1}"')
print(f'Sentence 2: "{sentence2}"')
print(f'Sentence 3: "{sentence3}"')

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

print('Comparing sentence 1 and 2 similarity:')
print(np.dot(embedding1, embedding2))
print('Comparing sentence 1 and 3 similarity:')
print(np.dot(embedding1, embedding3))
print('Comparing sentence 2 and 3 similarity:')
print(np.dot(embedding2, embedding3))


# Vector stores

from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/'

# Erase old database file in docs/chroma/ if any
import shutil
try:
    shutil.rmtree(persist_directory)
except FileNotFoundError:
    pass


vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())


# Similarity search
print('---')
question = "is there an email i can ask for help"
print(f'Question: "{question}"')

docs = vectordb.similarity_search(question, k=3)

print(len(docs))
print(docs[0].page_content)

# Save this so we can use it later
vectordb.persist()


# Failures modes
print('---')
question = "what did they say about pregnancy?"
print(f'Question: "{question}"')
docs = vectordb.similarity_search(question,k=5)
print(docs[0].page_content)
print(docs[1].page_content)


question = "what did they say about allowed therapies?"
print(f'Question: "{question}"')
docs = vectordb.similarity_search(question,k=5)
for doc in docs:
    print(doc.metadata)

print(docs[4].page_content)
