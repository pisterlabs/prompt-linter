# Import External Modules
import os

import openai

from dotenv import load_dotenv, find_dotenv

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback

def edital_analyzer(file_pdf):
    """Module to get informations about public contests."""
    docs = load_file(file_pdf)


def load_env_file():
    """Function to load the .env file."""
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ['OPENAI_API_KEY']

def load_file(file_pdf):
    """Function to load the pdf file."""
    loader = PyPDFLoader(file_pdf)
    documents = loader.load()
    return documents

def split_document(documents, 
                   model_name="gpt-3.5-turbo-16k", 
                   chunk_size=10000, 
                   chunk_overlap=200):
    """Function to split the document."""
    splitter = TokenTextSplitter(model_name==model_name)
    documents_splited = splitter.split_text(documents)
    return documents_splited