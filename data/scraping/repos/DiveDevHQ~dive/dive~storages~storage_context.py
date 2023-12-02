from typing import Optional, List
from dataclasses import dataclass
from langchain.vectorstores.base import VectorStore
from langchain.embeddings import SentenceTransformerEmbeddings
from dive.constants import DEFAULT_COLLECTION_NAME
from langchain.embeddings.base import Embeddings
import environ

env = environ.Env()
environ.Env.read_env()  # reading .env file
import os


@dataclass
class StorageContext:
    vector_store: VectorStore

    @classmethod
    def from_defaults(cls,
                      vector_store: Optional[VectorStore] = None,
                      embedding_function: Optional[Embeddings] = None):
        if not vector_store:
            PINECONE_API_KEY = env.str('PINECONE_API_KEY', default='') or os.environ.get('PINECONE_API_KEY', '')
            PINECONE_ENV = env.str('PINECONE_ENV', default='') or os.environ.get('PINECONE_ENV', '')
            PINECONE_INDEX_DIMENSIONS = env.str('PINECONE_INDEX_DIMENSIONS', default='512') or os.environ.get(
                'PINECONE_INDEX_DIMENSIONS', '512')
            if PINECONE_API_KEY:
                import_err_msg = (
                    "`pinecone` package not found, please run `pip install pinecone`"
                )
                try:
                    from dive.storages.vectorstores.pinecone import Pinecone
                    import pinecone
                    pinecone.init(
                        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
                        environment=PINECONE_ENV,  # next to api key in console
                    )
                    if not embedding_function:
                        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                    if DEFAULT_COLLECTION_NAME not in pinecone.list_indexes():
                        pinecone.create_index(
                            name=DEFAULT_COLLECTION_NAME,
                            metric="cosine",
                            dimension=int(PINECONE_INDEX_DIMENSIONS))
                    index = pinecone.Index(DEFAULT_COLLECTION_NAME)
                    vector_store = Pinecone(index, embedding_function, "text")

                except ImportError:
                    raise ImportError(import_err_msg)
            else:
                import_err_msg = (
                    "`chromadb` package not found, please run `pip install chromadb`"
                )
                CHROMA_SERVER = env.str('CHROMA_SERVER', default=None) or os.environ.get('CHROMA_SERVER', default=None)
                CHROMA_PORT = env.str('CHROMA_PORT', default=None) or os.environ.get('CHROMA_PORT', default=None)
                CHROMA_PERSIST_DIR = env.str('CHROMA_PERSIST_DIR', default='db') or os.environ.get('CHROMA_PERSIST_DIR',
                                                                                                   default='db')

                try:
                    import chromadb
                    from dive.storages.vectorstores.chroma import Chroma
                    from chromadb.config import Settings
                    if not embedding_function:
                        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                    chromadb_client=None

                    if CHROMA_SERVER:
                        CHROMA_PERSIST_DIR=None
                        chromadb_client=chromadb.HttpClient(host=CHROMA_SERVER, port=CHROMA_PORT)

                    vector_store = Chroma(
                        client=chromadb_client,
                        collection_name=DEFAULT_COLLECTION_NAME,
                        persist_directory=CHROMA_PERSIST_DIR,
                        embedding_function=embedding_function)

                except ImportError:
                    raise ImportError(import_err_msg)

        return cls(
            vector_store=vector_store
        )