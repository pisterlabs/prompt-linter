# import streamlit as st
from dotenv import load_dotenv
import pickle
# from PyPDF2 import PdfReader
# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma , FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
import os
import json

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import openai


openai.api_type = "open_ai"
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "NULL"
load_dotenv()

traningfolder = 'traning/'
pdfs = []
text_files = []

# load the document and split it into chunks
loader = PyPDFDirectoryLoader(traningfolder)
documents = loader.load()
#
# # split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load docs into Chroma DB
db = FAISS.from_documents(docs, embedding_function)

## saving the db
db = FAISS.from_documents(docs, embedding_function, persist_directory="./FAISS")
query = "who is naval"
docs = db.similarity_search(query)
print(docs[0].page_content)

# # save to disk
# db3 = FAISS(persist_directory="./mydb1", embedding_function=embedding_function)
# db3.get()
# query = "what is linked list"
# docs = db3.similarity_search(query)
# print(docs[0].page_content)
