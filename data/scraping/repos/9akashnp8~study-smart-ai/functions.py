from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import get_buffer_string

from decouple import config

from utils.constants import DB_DIR

def pdf_to_text_chunks(file_path: str, chunk_size: int = 500) -> "list[str]":
    """converts a pdf to chunks for embedding creation"""
    loader = PDFMinerLoader(file_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    docs = text_splitter.split_documents(document)
    return docs

def create_embeddings(docs: "list[str]", collection_name: str) -> None:
    """creates and persists embeddings to db"""
    embeddings = OpenAIEmbeddings(openai_api_key=config('OPENAI_API_KEY'))
    db = Chroma.from_documents(docs, embeddings, collection_name=collection_name, persist_directory=DB_DIR)
    db.persist()

def query_db(query: str, collection_name: str, n_results: int = 5) -> list:
    """query the vector db for similar text content"""
    embedding = OpenAIEmbeddings(openai_api_key=config('OPENAI_API_KEY'))
    db = Chroma(collection_name=collection_name, persist_directory=DB_DIR, embedding_function=embedding)
    results = db.similarity_search(query, k=n_results)
    return [result.page_content for result in results]

def get_chat_history(chat_history) -> str:
    if type(chat_history) == str:
        return chat_history
    return get_buffer_string(messages=chat_history)