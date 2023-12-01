import os
import logging
import sqlite3
import openai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
    PDFPlumberLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
)
from langchain.vectorstores import DeepLake
from dotenv import load_dotenv

load_dotenv()


class Library:
    def __init__(
        self,
        db_path="library.db",
        analyst_source=None,
        analyst_vs_path=None,
        snorkel_source=None,
        snorkel_vs_path=None,
    ):
        self.db_path = db_path
        self.conn = None
        self.c = None
        self.embeddings = OpenAIEmbeddings()

        # Knowledge sources

        # Analyst
        self.analyst_source = (
            analyst_source or "/Users/nick/Documents/Research/Gartner/test/"
        )
        self.analyst_vs_path = (
            analyst_vs_path or "./Deeplake/snkl_helper/research_data/"
        )
        self.analyst_vs = DeepLake(
            dataset_path=self.analyst_vs_path, embedding=self.embeddings
        )

        # Snorkel data
        self.snorkel_source = (
            snorkel_source or "/Users/nick/Git/snkl_assistant/data/snorkel_text_docs/"
        )
        self.snorkel_vs_path = analyst_vs_path or "./Deeplake/snkl_helper/snorkel_data/"
        self.snorkel_vs = DeepLake(
            dataset_path=self.snorkel_vs_path, embedding=self.embeddings
        )

    def _connect_to_db(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.c = self.conn.cursor()

            # Create table if doesn't exist
            self.c.execute(
                """
                CREATE TABLE IF NOT EXISTS added_docs 
                (id INTEGER PRIMARY KEY, doc_path TEXT)
                """
            )

    def close(self):
        if self.conn:
            self.conn.close()

    def process_directory(self, dir_path):
        total_files = sum([len(files) for _, _, files in os.walk(dir_path)])
        processed_files = 0

        for dirpath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                _, ext = os.path.splitext(filename)

                if ext == ".pdf":
                    file_path = os.path.join(dirpath, filename)
                    self.add_pdf(file_path)

                elif ext == ".docx":
                    file_path = os.path.join(dirpath, filename)
                    self.add_docx(file_path)

                processed_files += 1
                yield processed_files / total_files

    def document_exists(self, doc_path):
        self._connect_to_db()
        self.c.execute("SELECT 1 FROM added_docs WHERE doc_path=?", (doc_path,))
        return self.c.fetchone() is not None

    def insert_document_to_db(self, doc_path):
        self._connect_to_db()
        self.c.execute("INSERT INTO added_docs (doc_path) VALUES (?)", (doc_path,))
        self.conn.commit()

    def add_pdf(self, doc_path):
        if self.document_exists(doc_path):
            logging.info(f"Skipping already added doc: {doc_path}")
            return

        loader = PDFPlumberLoader(doc_path)
        doc = loader.load_and_split(
            text_splitter=TokenTextSplitter(chunk_size=200, chunk_overlap=0)
        )

        if not self.analyst_vs:
            logging.error("Analyst vector store not initialized")
            return

        self.analyst_vs.add_documents(doc)
        self.insert_document_to_db(doc_path)

        logging.info(f"Added new document: {doc_path}")

    def add_docx(self, doc_path):
        if self.document_exists(doc_path):
            logging.info(f"Skipping already added doc: {doc_path}")
            return

        loader = UnstructuredWordDocumentLoader(doc_path, mode="elements")
        doc = loader.load_and_split(
            text_splitter=TokenTextSplitter(chunk_size=200, chunk_overlap=0)
        )

        if not self.snorkel_vs:
            logging.error("Analyst vector store not initialized")
            return

        self.snorkel_vs.add_documents(doc)
        self.insert_document_to_db(doc_path)

        logging.info(f"Added new document: {doc_path}")

    # Ensure to close the connection when done
    def __del__(self):
        self.close()
