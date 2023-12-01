from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from kmai.ports.ivectorstore_helper import IVectorStoreHelper


class FAISSHelper(IVectorStoreHelper):
    def create_vectorstore(self, docs):
        embeddings = OpenAIEmbeddings()
        return FAISS.from_documents(docs, embeddings)

    def read_vectorstore(self, path: str):
        embeddings = OpenAIEmbeddings()
        try:
            db = FAISS.load_local(path, embeddings)
        except:
            return None
        return db

    def write_vectorstore(self, db: FAISS, path: str):
        db.save_local(path)
        return True

    def similarity_search(self, db: FAISS, text: str, k: int):
        return db.similarity_search(text, k)

    def add_doc_to_vectorstore(self, db, docs):
        db.add_documents(documents=docs)
        return db
