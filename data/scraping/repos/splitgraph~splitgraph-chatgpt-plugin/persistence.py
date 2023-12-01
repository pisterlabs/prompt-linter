from math import ceil, sqrt
from typing import List, Optional, Tuple
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore
from langchain.vectorstores import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
import sqlalchemy
from sqlalchemy.orm import Session

CLOUDSQL_PG_CONN_STR = "postgresql+pg8000://"


class PGVectorReuseConnection(PGVector):

    _conn: Optional[sqlalchemy.engine.Connection] = None

    def __init__(self, connection: sqlalchemy.engine.Connection, *args, **kwargs):
        self._conn = connection
        super().__init__(*args, connection_string="thisisignored", **kwargs)

    def connect(self) -> sqlalchemy.engine.Connection:
        if self._conn:
            return self._conn
        return super().connect()


def create_pgvector_index(db: PGVector, max_elements: int):
    create_index_query = sqlalchemy.text(
        "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_idx "
        "ON langchain_pg_embedding "
        "USING ivfflat (embedding vector_cosine_ops) "
        # from: https://supabase.com/blog/openai-embeddings-postgres-vector#indexing
        # "A good starting number of lists is 4 * sqrt(table_rows)"
        "WITH (lists = {});".format(ceil(4 * sqrt(max_elements)))
    )
    # Execute the queries
    try:
        with Session(db._conn) as session:
            # Create the HNSW index
            session.execute(create_index_query)
            session.commit()
        print("PGVector extension and index created successfully.")
    except Exception as e:
        print(f"Failed to create PGVector extension or index: {e}")


def get_embedding_store_pgvector(
    connection: sqlalchemy.engine.Connection,
    collection: str,
    openai_api_key: str,
    max_elements: int = 100000,
) -> VectorStore:
    # description: https://supabase.com/blog/openai-embeddings-postgres-vector

    db = PGVectorReuseConnection(
        connection,
        # MyPy chokes on this, see: https://github.com/langchain-ai/langchain/issues/2925
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),  # type: ignore
        collection_name=collection,
        distance_strategy=DistanceStrategy.COSINE,
    )
    # create index
    create_pgvector_index(db, max_elements)
    return db


def find_repos(vstore: VectorStore, query: str, limit=4) -> List[Tuple[str, str]]:
    results = vstore.similarity_search_with_score(query, limit)
    # sort by relevance, returning most relevant repository first
    results.sort(key=lambda a: a[1], reverse=True)
    # deduplicate results
    return list(
        set(
            [(r[0].metadata["namespace"], r[0].metadata["repository"]) for r in results]
        )
    )


def connect(connection_string: str) -> sqlalchemy.engine.Connection:
    if connection_string == CLOUDSQL_PG_CONN_STR:
        from .db_cloudsql import connect_with_connector

        return connect_with_connector().connect()
    engine = sqlalchemy.create_engine(connection_string)
    return engine.connect()
