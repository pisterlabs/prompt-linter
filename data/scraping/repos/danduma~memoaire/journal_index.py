import json
import os
from typing import List, Iterator

from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from formats.journal_db import JournalDatabase
from utils import get_compute_device

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_document_from_json(data: dict) -> Document:
    """Create a Langchain Document from a JSON dict."""
    page_content = data['body']
    metadata = {"date": data['date'], "title": data['title']}
    return Document(page_content=page_content, metadata=metadata)


class SQLiteDocumentLoader(BaseLoader):
    """A loader that creates a Langchain Document from each json dict in a jsonl file."""

    def __init__(self, file_path: str):
        self.db = JournalDatabase(file_path)

    def load(self) -> List[Document]:
        """Load data into Document objects."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        for entry in self.db.iterate_entries():
            yield create_document_from_json(entry)


class JSONLDocumentLoader(BaseLoader):
    """A loader that creates a Langchain Document from each json dict in a jsonl file."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load data into Document objects."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        with open(self.file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                doc = create_document_from_json(data)
                yield doc


class JournalIndex:
    def __init__(self, db_path, vector_dir, settings={}):
        self.db_path = db_path

        text_embedding_model = settings.get('text_embedding_model', "BAAI/bge-base-en")
        event_embedding_model = settings.get('text_embedding_model', "BAAI/bge-base-en")
        vector_db_provider = settings.get('vector_db_provider', "faiss")

        model_kwargs = {'device': get_compute_device()}
        encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
        self.text_embedding = HuggingFaceBgeEmbeddings(model_name=text_embedding_model, model_kwargs=model_kwargs,
                                                       encode_kwargs=encode_kwargs)
        # self.events_embedding = HuggingFaceBgeEmbeddings(model_name=event_embedding_model, model_kwargs=model_kwargs,
        #                                                encode_kwargs=encode_kwargs)

        self.vector_dir = vector_dir
        if vector_db_provider.lower() == "faiss":
            vectordb_text_path = os.path.join(self.vector_dir, 'text')
            # vectordb_event_path = os.path.join(self.vector_dir, 'events')

            if os.path.exists(vectordb_text_path):
                self.text_vectorstore = FAISS.load_local(vectordb_text_path, self.text_embedding)
            else:
                self.text_vectorstore = FAISS.from_documents([Document(page_content='', metadata={})], self.text_embedding)

            # if os.path.exists(vectordb_text_path):
            #     self.event_vectorstore = FAISS.load_local(vectordb_events_path, self.events_embedding)
        else:
            raise NotImplementedError

    def create_index(self):
        loader = SQLiteDocumentLoader(self.db_path)
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(raw_documents)

        self.text_vectorstore = FAISS.from_documents(documents, self.text_embedding)
        full_path = os.path.join(self.vector_dir, 'text')

        self.text_vectorstore.save_local(full_path)

    def delete_index(self):
        if self.text_vectorstore:
            self.text_vectorstore = FAISS.from_documents([Document(page_content='', metadata={})], self.text_embedding)
        full_path = os.path.join(self.vector_dir, 'text')
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except PermissionError:
                pass

    def as_retriever(self, search_kwargs):
        return self.text_vectorstore.as_retriever(search_kwargs=search_kwargs)
