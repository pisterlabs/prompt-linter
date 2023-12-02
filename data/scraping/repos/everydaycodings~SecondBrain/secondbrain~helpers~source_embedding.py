from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from helpers.utils import load_embedding_model
import streamlit as st
import tempfile, os


class ChatSourceEmbedding:

    def __init__(self) -> None:
        pass

    def embedding_chat(self, model_name, device, persist_directory, prompt, search_kwargs):

        embedding = load_embedding_model(model_name, device)

        vectordb = Chroma(persist_directory="SecondBrain/secondbrain/database/{}".format(persist_directory), 
                  embedding_function=embedding)
        
        retriever = vectordb.as_retriever(search_kwargs={"k": search_kwargs})

        docs = retriever.get_relevant_documents(prompt)

        return docs