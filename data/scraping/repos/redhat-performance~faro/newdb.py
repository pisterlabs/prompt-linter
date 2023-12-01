from flask import Flask, session, request, jsonify
from flask_cors import CORS, cross_origin
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from flask_session import Session
from langchain.chat_models import ChatOpenAI
import openai
import chromadb
from chromadb.config import Settings
import time
from langchain.embeddings import GPT4AllEmbeddings
from gpt4all import GPT4All

documents = DirectoryLoader("output").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)
embeddings = GPT4AllEmbeddings()


client = chromadb.PersistentClient(path="persist/")


vectorstore = Chroma.from_documents(
    documents=documents,
    client=client, collection_name="newcol", embedding_function=embeddings
)
