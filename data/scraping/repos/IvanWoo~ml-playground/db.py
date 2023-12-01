import os

from langchain.vectorstores import Milvus
from langchain.vectorstores import PGVector

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

PG_COLLECTION_NAME = "langchain_pgvector_collection"
PG_CONNECTION_STRING = (
    "postgresql+psycopg2://postgres:demo_password@localhost:5432/postgres"
)

DB_BACKEND = os.environ.get("DB_BACKEND", "pgvector")


def from_documents(
    docs, embeddings, db: str = DB_BACKEND, reset: bool = False, **kwargs
):
    """
    langchain vectorstores.from_documents wrapper extended with
    - db backend options
    - unified reset flag
    """
    print(f"db backend is {db}")
    db = db.lower()
    if db == "milvus":
        return Milvus.from_documents(
            docs,
            embeddings,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            drop_old=reset,
            **kwargs,
        )
    elif db == "pgvector":
        return PGVector.from_documents(
            embedding=embeddings,
            documents=docs,
            collection_name=PG_COLLECTION_NAME,
            connection_string=PG_CONNECTION_STRING,
            pre_delete_collection=reset,
            **kwargs,
        )
    else:
        raise ValueError(f"unsupported {db=}")
