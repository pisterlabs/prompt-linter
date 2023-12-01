from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from fastapi import HTTPException


async def load_data(path, file_type):
    if file_type == 'application/pdf':
        loader_first = PyPDFLoader(path)
        pages = loader_first.load_and_split()
        return pages
    elif file_type == 'text/plain':
        loader_first = TextLoader(path, encoding='utf-8')
        pages = loader_first.load_and_split()
        return pages
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        loader_first = Docx2txtLoader(path)
        pages = loader_first.load()
        return pages
    else:
        raise ValueError("File type %s is not a valid file type" % file_type)


async def lazy_load_data(path, file_type):
    if file_type == 'application/pdf':
        loader_first = PyPDFLoader(path)
        pages = loader_first.load()
        return pages
    elif file_type == 'text/plain':
        loader_first = TextLoader(path, encoding='utf-8')
        pages = loader_first.load()
        return pages
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        loader_first = Docx2txtLoader(path)
        pages = loader_first.load_and_split()
        return pages
    else:
        raise ValueError("File type %s is not a valid file type" % file_type)
