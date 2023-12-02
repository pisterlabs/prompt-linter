import faiss
import os
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


class Vdb:
    def __init__(self, base_path, name, separator=' ', chunk_size=1500, chunk_overlap=0):
        self.base_path = base_path
        self.name = name
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def index_name(self):
        return f'{self.base_path}/{self.name}.idx'

    def exists(self):
        return os.path.exists(self.index_name()) and os.path.exists(self.pickle_name())

    def pickle_name(self):
        return f'{self.base_path}/{self.name}.pkl'

    def build(self, data):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = []
        if isinstance(data, str):
            docs = splitter.split_text(data)
        else:
            docs = [t.page_content for t in splitter.split_documents(data)]

        store = FAISS.from_texts(docs, OpenAIEmbeddings())
        faiss.write_index(store.index, self.index_name())
        store.index = None
        with open(self.pickle_name(), 'wb') as f:
            pickle.dump(store, f)

    def load(self):
        with open(self.pickle_name(), 'rb') as f:
            store = pickle.load(f)
        store.index = faiss.read_index(self.index_name())
        return store
