import os

#Env Mgt
from dotenv import load_dotenv
from pypdf import PdfReader
from torch import embedding

api_key = os.getenv('OPENAI_API_KEY')
data_conn = os.getenv('DATA_CONNECTION_STRING', './data/database.db')
data_processed= os.getenv('DATA_PROCESSED_DIR', './source/processed')
data_pre_processed= os.getenv('DATA_PRE_PROCESSED_DIR', './source/preprocessed')
data_source= os.getenv('DATA_SOURCE_DIR', './source')
data_directory= os.getenv('DATA_DIRECTORY', './data')
log_level = 'INFORMATION'

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader

def load_data():
    loader = PyPDFLoader(f'{data_source}/NAP NPP Final with print outs 09.06.16.pdf')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embedding = OpenAIEmbeddings()
    
    vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=data_directory)
    vectordb.persist()

if __name__ == '__main__':
    load_data()

