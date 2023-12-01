"""
Abstractions for (fixed) document databases
"""
import logging
import os
from typing import List

from langchain.document_loaders import PythonLoader, TextLoader
from langchain.embeddings.base import Embeddings
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma

log = logging.getLogger(__name__)


class DocumentDatabase:
    def __init__(self, name: str, documents: List[Document]):
        self.name = name
        self.documents = documents


class SingleTextFileDocumentDatabase(DocumentDatabase):
    def __init__(self, name: str, textfile: str):
        super().__init__(name, TextLoader(textfile).load())


class PythonDocumentDatabase(DocumentDatabase):
    def __init__(self, name: str, src_root: str):
        documents = []
        for root, dirs, files in os.walk(src_root):
            for fn in files:
                fn: str
                if fn.endswith(".py"):
                    pypath = os.path.join(root, fn)
                    documents.extend(PythonLoader(pypath).load())
        super().__init__(name, documents)


class VectorDatabase:
    DB_ROOT_DIRECTORY = "vectordb"

    def __init__(self, name: str, doc_db: DocumentDatabase, splitter: TextSplitter, embedding_function: Embeddings,
            load=True):
        self.name = name
        self.embedding_function = embedding_function
        self.doc_db = doc_db
        self.splitter = splitter
        self.db = self._get_or_create_db(load=load)

    def _db_directory(self) -> str:
        return f"{self.DB_ROOT_DIRECTORY}/{self.name}"

    def _get_or_create_db(self, load=True) -> Chroma:
        if load and os.path.exists(os.path.join(self._db_directory(), "chroma-embeddings.parquet")):
            db = Chroma(embedding_function=self.embedding_function, persist_directory=self._db_directory())
        else:
            texts = self.splitter.split_documents(self.doc_db.documents)
            log.info(f"Documents were split into {len(texts)} sub-documents")

            db = Chroma.from_documents(texts, self.embedding_function, persist_directory=self._db_directory())
            db.persist()
        return db

    def retriever(self) -> BaseRetriever:
        return self.db.as_retriever()
