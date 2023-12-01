from langchain.document_loaders import UnstructuredURLLoader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.chat_models import ChatAnthropic
import os

load_dotenv()
chat = ChatAnthropic()

def get_text_from_link(link):
    loaders = UnstructuredURLLoader(urls=[link])
    data = loaders.load()
    return data

def split_text(data):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    return docs

def get_embeddings(docs): # will get embeddings from docs and then store them in a vector database
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory='vectorstore'
    )
    vectorstore.persist()
    return vectorstore


def getting_ans(query, vectorstore):
    llm = ChatAnthropic()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), verbose=True, chain_type="stuff")
    result = chain({"query": query}, return_only_outputs=True)
    return result['result']


def main():
    print('0')
    query = st.session_state.ques
    data = get_text_from_link(st.session_state.linkkk) 
    print('1')
    docs = split_text(data)
    print('2')
    vectorstore = get_embeddings(docs)
    print('3')
    ans = getting_ans(query, vectorstore)
    print('4')
    print(ans)


    st.session_state.history.append({"message": query, "is_user": True})
    st.session_state.history.append({"message": ans, "is_user": False})

