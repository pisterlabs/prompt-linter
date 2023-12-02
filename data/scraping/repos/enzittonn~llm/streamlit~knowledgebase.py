import os
from typing import Optional

from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter



##### External imports #####
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

#####


CHROMA_DB_DIRECTORY='data/chroma'
DOCUMENT_SOURCE_DIRECTORY='data/files/Fastighet.pdf'
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=CHROMA_DB_DIRECTORY,
    anonymized_telemetry=False
)
TARGET_SOURCE_CHUNKS=4
CHUNK_SIZE=500
CHUNK_OVERLAP=50
HIDE_SOURCE_DOCUMENTS=False

class KnowledgeBase:
    def __init__(self, pdf_source_folder_path: str) -> None:
        """
        Loads pdf and creates a Knowledge base using the Chroma
        vector DB.
        Args:
            pdf_source_folder_path (str): The source folder containing 
            all the pdf documents
        """
        self.pdf_source_folder_path = pdf_source_folder_path

    def load_directory(self):
        loader = DirectoryLoader(
            self.pdf_source_folder_path
        )
        loaded_pdfs = loader.load()
        return loaded_pdfs
    
    # Works - returns
    def langchain_pdf_loader(self):
        loader = PyPDFLoader(self.pdf_source_folder_path)
        pages = loader.load()
        return pages

    def split_documents(
        self,
        loaded_docs,
    ):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs




    # Does not work - fix
    def convert_document_to_FAISS_embeddings(
        self, chunked_docs, embedder
    ):
        faiss_db = FAISS.afrom_documents(self, chunked_docs, embedder)
        faiss_db.save_local('data/faiss/faiss_index')
        
        return faiss_db
    


    
    
    # # NO Working
    # def convert_document_to_chroma_embeddings(
    #     self, chunked_docs, embedder
    # ):
    #     vector_db = Chroma.from_documents(
    #         persist_directory=CHROMA_DB_DIRECTORY,
    #         embedding=embedder,
    #         documents=chunked_docs,
    #         client_settings=CHROMA_SETTINGS,
    #     )

    #     vector_db.add_documents(chunked_docs)
    #     vector_db.persist()
    #     return vector_db





    def return_retriever_from_persistant_vector_db(
        self, embedder
    ):
        if not os.path.isdir(CHROMA_DB_DIRECTORY):
            raise NotADirectoryError(
                "Please load your vector database first."
            )
        
        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=embedder,
            client_settings=CHROMA_SETTINGS,
        )

        return vector_db.as_retriever(
            search_kwargs={"k": TARGET_SOURCE_CHUNKS}
        )

    def initiate_document_injetion_pipeline(self):
        loaded_pdfs = self.langchain_pdf_loader()
        chunked_documents = self.split_documents(loaded_docs=loaded_pdfs)
        
        print("=> PDF loading and chunking done.")

        embeddings = OpenAIEmbeddings()
        vector_db = self.convert_document_to_chroma_embeddings(
            chunked_docs=chunked_documents, embedder=embeddings
        )

        print("=> vector db initialised and created.")
        print("All done")