import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
import tempfile

load_dotenv('.env')

st.set_option('deprecation.showfileUploaderEncoding', False)

documents = []

st.title("DocBot - Document Search and Question Answering")

uploaded_files = st.file_uploader("Upload document(s)", type=["pdf", "docx", "doc", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.write("Uploading documents...")

    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(documents)

    # Convert the document chunks to embedding and save them to the vector store
    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
    vectordb.persist()

    # Create the Q&A chain
    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
        retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False
    )

    chat_history = []

    st.write('Documents have been uploaded and processed. You can now start asking questions.')

    while True:
        query = st.text_input("Prompt:", key="question_input")
        if st.button("Submit"):
            if query == "exit" or query == "quit" or query == "q" or query == "f":
                st.write('Exiting')
                sys.exit()
            if query:
                result = pdf_qa(
                    {"question": query, "chat_history": chat_history})
                st.write("Answer: " + result["answer"])
                chat_history.append((query, result["answer"]))
