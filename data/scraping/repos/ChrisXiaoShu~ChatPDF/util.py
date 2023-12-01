import os

from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


# function to load only .txt or .pdf file and will return error when file not found or can't not load
def load_file(file_path) -> TextLoader or PyPDFLoader:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File is not a file: {file_path}")
    if file_path.endswith(".txt"):
        return TextLoader(file_path)
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path)
    raise ValueError(f"File type not supported: {file_path}")

# function pass Loadder and return splitter texts
def split_text(loader: TextLoader or PyPDFLoader) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return loader.load_and_split(splitter)

# create emmbeding vectorstore by using opentAIEmbedding model from given text
def create_embedding_vectorstore(texts: list) -> Chroma:
    embedding_model = OpenAIEmbeddings()
    return Chroma.from_documents(texts, embedding_model)