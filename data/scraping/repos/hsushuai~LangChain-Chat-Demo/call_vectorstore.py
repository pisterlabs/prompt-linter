from typer import Option

from embeddings import get_embedding
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from configs import PERSIST_DIR


def get_vectordb(embedding, files: Option = None, persist_directory: str = PERSIST_DIR) -> Chroma:
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    if files is not None:
        loader = UnstructuredFileLoader(files)
        docs = loader.load()
        vectordb = vectordb.from_documents(embedding=embedding, documents=docs)
    return vectordb
