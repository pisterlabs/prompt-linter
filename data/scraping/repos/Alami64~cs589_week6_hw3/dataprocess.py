from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import os
import openai
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

openai.api_key  = os.environ['OPENAI_API_KEY']



def load_vectorstore(file,k=2):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)
    # define retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever


def return_answer(temperature, model,memory, retriever,chaintype="stuff"):
    qa = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(model_name=model, temperature=temperature, max_tokens=300),
        chain_type=chaintype,
        retriever=retriever,
        memory=memory
    )
    return qa



