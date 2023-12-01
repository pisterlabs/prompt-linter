import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document


class KnowledgeBase:
    def __init__(self, embeddings: Embeddings, persistence_folder: os.PathLike | None = None, index_name: str = 'index') -> None:
        self.index_name = index_name
        self.persistence_folder = persistence_folder
        self.save = persistence_folder is not None

        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        if self.save:
            if os.path.exists(self.persistence_folder):
                if os.path.exists(os.path.join(self.persistence_folder, f'{self.index_name}.faiss')):
                    self.db = FAISS.load_local(self.persistence_folder, self.embeddings, self.index_name)
                else:
                    self.db = None
            else:
                os.makedirs(self.persistence_folder)  # for future saving
                self.db = None
        else:
            self.db = None

    def remove_documents(self) -> None:
        self.db = None

    def add_document(self, file_path_or_url: os.PathLike | str) -> None:
        loader = PyPDFLoader(file_path_or_url)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)

        if self.db is None:
            self.db = FAISS.from_documents(splitted_documents, self.embeddings)
        else:
            self.db.add_documents(splitted_documents)

        if self.save:
            self.db.save_local(self.persistence_folder, self.index_name)

    def query(self, query: str) -> list[Document]:
        return [] if self.db is None else self.db.similarity_search(query, k=4)
