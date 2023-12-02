# pyhton
import os
from enum import Enum

# local
from app_managers.system_util import IncompleteSetupException

# langchain
# Vector store
from langchain.vectorstores import Chroma

# implementation


class RetrieverSearchType(Enum):
    similarity = 1
    mmr = 2
    similarity_score_threshold = 3


class VDBType(Enum):
    chroma = 1
    faiss = 2


class VectorDBManager:
    def __init__(self, db_type=VDBType.chroma, embedding_model=None, db_dir=None, documents=None):
        self.db_type = db_type
        self.embedding_model = embedding_model
        self.db_dir = db_dir
        self.vdb = self.get_vector_store(
            embedding_model=embedding_model, documents=documents)
        self.retriever = self.get_retriever() if self.vdb else None

    def get_db_size(self):
        db_size = 0
        emb_size = 0
        if self.vdb:
            db_size = self.vdb._collection.count()
            if db_size > 0:
                emb_size = len(self.vdb._collection.peek(
                    limit=1)['embeddings'][0])
        return db_size, emb_size

    def get_vector_store(self, embedding_model, documents):
        # Supplying a persist_directory will store the embeddings on disk
        if not self.embedding_model:
            raise IncompleteSetupException(
                "Embedding model not set. Please set the embedding model.")
        # if self.is_persistent and not self.db_dir:
        #     raise IncompleteSetupException(
        #         "db dir not set. Persitent db needs dir. Please set the db dir.")

        vectordb = None
        if self.db_type == VDBType.chroma:
            if documents:
                vectordb = Chroma.from_documents(persist_directory=self.db_dir,
                                                 embedding=self.embedding_model,
                                                 documents=documents)
            else:
                vectordb = Chroma(persist_directory=self.db_dir,
                                  embedding_function=self.embedding_model)

        return vectordb

    def add_documents_to_store(self, documents):
        if self.vdb:
            return self.vdb.add_documents(documents)

    def add_texts_to_store(self, texts):
        if self.vdb:
            return self.vdb.add_texts(texts)

    def get_retriever(self, search_type="similarity", max_match=3, score_threshold=0.75):
        if not self.vdb:
            raise IncompleteSetupException(
                "Vector DB not set. Please set Vector DB")
        retriever = self.vdb.as_retriever(
            search_kwargs={"k": max_match, 'score_threshold': score_threshold}, search_type=search_type, )
        return retriever

    def delete_vdb_dir(self):
        if self.is_persistent and self.db_dir:
            os.system(f"rm -rf {self.db_dir}")

    @staticmethod
    def get_default_persistent_dir():
        return './vdb/chroma/'
