import re
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredURLLoader, SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
# A set of loaders for pdf files, text files and website links
# Each of them returns a Document object

def load_text_file(file_path: str) -> Document:
    """ Loads a text file and returns a Document object.

    Args:
        file_path: path of the text file.

    Returns:
        A Document object.
    """
    doc = TextLoader(file_path, encoding='utf-8').load()[0]
    return doc

def load_pdf_file(file_path: str) -> List[Document]:
    """ Loads a pdf file and returns a Document object.

    Args:
        file_path: path of the pdf file.

    Returns:
        A Document object.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def load_website(url: str) -> Document:
    """ Loads a website and returns a Document object.

    Args:
        url: Url of the website.

    Returns:
        A Document object.
    """
    return UnstructuredURLLoader(urls=[url]).load()
