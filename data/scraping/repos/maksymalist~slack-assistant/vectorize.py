from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

import requests
import os


def download_pdf_from_url(url, file_name, save_directory):
    response = requests.get(url)
    
    if response.status_code == 200:
        
        # Set the save path
        save_path = os.path.join(save_directory, file_name)
        
        # Save the PDF to the specified directory
        with open(save_path, "wb") as file:
            file.write(response.content)
        
        print(f"PDF downloaded successfully and saved as: {save_path}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")


def vectorize_document(path):
    loader = PyMuPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    persist_directory = "./storage"
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=texts, 
                                    embedding=embeddings,
                                    persist_directory=persist_directory)
    vectordb.persist()
    


def get_db():
    persist_directory = "./storage"
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    return vectordb