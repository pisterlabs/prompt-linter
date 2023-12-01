import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pages.front.htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
import os

load_dotenv()

codigo_civil = "./constitucion/codigo_civil_colombia.pdf"
codigo_comercio = "./constitucion/codigo_comercio.pdf"
codigo_penal = "./constitucion/codigopenal_colombia.pdf"

pdf_docs_law = [codigo_civil, codigo_comercio, codigo_penal]

def get_pdf_text(pdf_docs_law):
    text = ""
    for pdf in pdf_docs_law:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory_law = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain_law = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory_law
    )
    return conversation_chain_law

def main():
    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    if "conversation_law" not in st.session_state:
        st.session_state.conversation_law = None
    if "chat_history_law" not in st.session_state:
        st.session_state.chat_history_law = None

    st.header("Asistente virtual Juris Bot :books:")
    user_question_law = st.text_input("Pregunta cualquier dato sobre la constitución Colombiana:")
    if user_question_law:
        handle_userinput(user_question_law)


def handle_userinput(user_question_law):
    if st.session_state.conversation_law is None:
        raw_text_law = get_pdf_text(pdf_docs_law)
        text_chunks_law = get_text_chunks(raw_text_law)
        vectorstore_law = get_vectorstore(text_chunks_law)
        st.session_state.conversation_law = get_conversation_chain(vectorstore_law)


    response_law = st.session_state.conversation_law({'question': user_question_law})
    st.session_state.chat_history_law = response_law['chat_history']

    for i, message in enumerate(st.session_state.chat_history_law):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
              
if __name__ == '__main__':
    main()