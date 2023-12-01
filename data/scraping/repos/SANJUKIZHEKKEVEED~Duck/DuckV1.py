import os
import sys
import sqlite3
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

# Create a Streamlit app
st.title("🦣Doc BOT")

# Relative paths to the 'docs' and 'data' folders in the GitHub repository
DOCS_DIRECTORY = "docs"
PERSIST_DIRECTORY = "db"

# Create a place to input the OpenAI API Key
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Document loading and processing
documents = []

if openai_api_key:
    # Set the OpenAI API key if it has been provided
    os.environ['OPENAI_API_KEY'] = openai_api_key

    for file in os.listdir(DOCS_DIRECTORY):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DOCS_DIRECTORY, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = os.path.join(DOCS_DIRECTORY, file)
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = os.path.join(DOCS_DIRECTORY, file)
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(documents)

    # Remove the invalid keyword argument
    embedding = OpenAIEmbeddings()

    # Create the vector database
    vectordb = Chroma.from_documents(documents, embedding=embedding, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()

    # Create the chat model
    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
        vectordb.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True
    )

    chat_history = []
    st.write("Document BOT: Work smarter, not harder, with your documents 🪄")

    # Allow the user to input a prompt
    query = st.text_input("Query Corner:")

    # Add a submit button with a spinner
    if st.button("Submit"):
        with st.spinner("Answering..."):
            if query:
                result = pdf_qa({"question": query, "chat_history": chat_history})
                st.write("Answer:", result["answer"])
                chat_history.append((query, result["answer"]))
