import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import WebBaseLoader
import os
from pathlib import Path

# Sidebar contents
with st.sidebar:
    st.title('💬 LLM chatbot 📄')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    "[View source code](https://github.com/Timothy1102/company-assistant-bot)"
 
load_dotenv()

def generate_response(url, question):
    if url is not None:
        loader = WebBaseLoader(url)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings()
        # Create a vector store from documents
        db = FAISS.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        llm = OpenAI(temperature=0.4)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
        return qa.run(question)
 
def main():
    st.header("💬 URL Q&A chatbot")
 
    # Accept user question
    url = st.text_input("URL:")
    question = st.text_input(
        "Ask your question:",
        placeholder="Summarize this page for me",
        disabled=not url,
    )

    if question and url:
        response = generate_response(url, question)
        st.write(response)
 
if __name__ == '__main__':
    main()
