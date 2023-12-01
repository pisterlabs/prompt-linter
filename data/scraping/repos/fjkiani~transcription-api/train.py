import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from secret_config import LoadSecrets


# Loading Secrets:
s_in = LoadSecrets()
s_in.load_secret()

# Setting Up Document Loader:
loader = DirectoryLoader(
    "transcript_folder/",
    glob="**/*.txt",
    loader_cls=TextLoader,
)

documents = loader.load()

# Text Splitting Configuration:

text_splitter = TokenTextSplitter(        
    chunk_size = 100,
    chunk_overlap  = 50
)


texts = text_splitter.split_documents(documents)

# Setting Up Embeddings and Vector Database:
persist_directory = "transcript_db"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=1)


vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory

)
vectordb.persist()

# Querying and Matching Documents:
query = "when did neil Armstrong flew the vehicle?"
matching_docs = vectordb.similarity_search(query, k=1)
print(matching_docs)


# This code snippet shows how to set up a document retrieval system using pre-trained embeddings and vector-based similarity searches. It highlights essential tasks like document loading, text splitting, embedding computation, and query matching, combining these elements to build a powerful and flexible information retrieval system