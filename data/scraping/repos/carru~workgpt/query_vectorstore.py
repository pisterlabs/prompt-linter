#!/usr/bin/env python3
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from constants import *

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
collection = db.get()

# print([metadata['source'] for metadata in collection['metadatas']])
print(len(collection['metadatas']))