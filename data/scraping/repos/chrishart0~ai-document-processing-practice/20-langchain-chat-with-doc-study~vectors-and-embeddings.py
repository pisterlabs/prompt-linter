import os
import openai
import shutil

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("data/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("data/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("data/MachineLearning-Lecture03.pdf")
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


##################
### Embeddings ###
##################

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

# Meausre the dot product between vectors to compare how similar they are
import numpy as np
print(np.dot(embedding1, embedding2))
print(np.dot(embedding1, embedding3))

####################
### Vector Store ###
####################

# Choose chroma becasue it is light wright and in memory
# ! pip install chromadb
from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'

# Try to remove the tree; if it fails, throw an error using try...except.
if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
    print("Deleting old choma data store")
    shutil.rmtree(persist_directory)

# Create the vector store
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())

print("")
print("Query VectorDB")
# Start to ask the vector store quetions, this will look for
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
print(len(docs))
print(docs[0].page_content)

# Try another question
print("")
question = "what did they say about matlab?"
docs = vectordb.similarity_search(question,k=5)
print(docs[0])

# persist the Vector DB
vectordb.persist()