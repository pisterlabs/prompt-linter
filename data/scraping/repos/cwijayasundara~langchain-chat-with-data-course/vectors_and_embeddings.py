import os
import openai
import sys

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    PyPDFLoader("docs/cs229_lectures/machinelearning-lecture01.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

splits = text_splitter.split_documents(docs)
print("There are " + str(len(splits)) + " splits in the list of splits")

# Embedings
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

np.dot(embedding1, embedding2)
np.dot(embedding1, embedding3)
np.dot(embedding2, embedding3)

# Vectorstores
from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

# Similarity Search
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
len(docs)
docs[0].page_content
vectordb.persist()

# Failure modes
question = "what did they say about matlab?"
docs = vectordb.similarity_search(question,k=5)

print(docs[0].page_content)

question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(question,k=5)
for doc in docs:
    print(doc.metadata)
print(docs[0].page_content)

print('There are ', vectordb._collection.count(), ' documents in the vectorstore')

