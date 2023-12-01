import os

from langchain.document_loaders import PyPDFLoader


def load_pdf(pdf_file):
    loader = PyPDFLoader(file_path=pdf_file)
    pages = loader.load_and_split()

    os.remove(pdf_file)

    return pages
