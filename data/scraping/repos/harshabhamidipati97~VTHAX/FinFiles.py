import streamlit as st

import openai
import os

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from dotenv import load_dotenv


def process_llm_response(llm_response):
    return llm_response['result']


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
turbo_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                    temperature=0,
                    model_name='gpt-4'
                    )


def handle(file, prompt):
    loader = UnstructuredFileLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=texts,
                                    embedding=embedding,
                                    persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None

    vectordb = Chroma(persist_directory=persist_directory,
                    embedding_function=embedding)

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            return_source_documents=True)
    llm_response = qa_chain(prompt)
    return process_llm_response(llm_response)

st.title(':blue[FINMAP] : Navigating the World of Numbers')

st.header('FinFiles : Know Your Document Inside Out')

uploaded_file = st.file_uploader('Upload a PDF file', type='pdf')
if uploaded_file is not None:
    temp_dir = '/tmp/'
    temp_file_path = os.path.join('temp/', uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    # Create the 'local_storage' directory if it doesn't exist
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

    prompt = st.text_area('Enter your query about the PDF file', height=200)
    
    if st.button("Answer"):
        with st.spinner('Writing your answer...'):
            st.write(handle(temp_file_path, prompt))
        
