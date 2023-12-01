import chromadb
from pydantic import BaseModel
import streamlit as st
from typing import List
from langchain.schema import Document
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

CHROMA_DB_DIRECTORY = "./db/chroma_db"
COLLECTION_NAME = "schema_collection"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


class Config(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 0
    docs_dir: str = "./docs/"
    docs_glob: str = "**/*.md"


class DocumentProcessor:
    def __init__(self, config: Config):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIRECTORY)
        self.loader = DirectoryLoader(config.docs_dir, glob=config.docs_glob)
        self.text_splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def load_documents(self) -> List[Document]:
        try:
            return self.loader.load()
        except Exception as e:
            # TODO: Appropriate error handling
            raise e

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)

    def process(self) -> Chroma:
        documents = self.load_documents()
        docs = self.split_documents(documents)
        db = Chroma.from_documents(
            docs,
            self.embeddings,
            client=self.client,
            collection_name=COLLECTION_NAME,
        )
        return db


def run():
    print("Starting document processing...")
    config = Config()
    processor = DocumentProcessor(config)
    result = processor.process()
    print("Document processing completed.")
    return result


if __name__ == "__main__":
    run()
