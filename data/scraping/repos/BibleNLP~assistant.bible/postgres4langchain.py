"""Implemetations for vectordb interface for postgres with vector store"""
import math
import os
from typing import List, Optional
from langchain.schema import Document as LangchainDocument
from langchain.schema import BaseRetriever
from core.vectordb import VectordbInterface
from core.embedding import EmbeddingInterface
import schema
from custom_exceptions import PostgresException, GenericException
import numpy as np

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from log_configs import log

# pylint: disable=too-few-public-methods, unused-argument, too-many-arguments, R0801
QUERY_LIMIT = os.getenv("POSTGRES_DB_QUERY_LIMIT", "10")
MAX_COSINE_DISTANCE = os.getenv("POSTGRES_MAX_COSINE_DISTANCE", "0.1")


class Postgres(
    VectordbInterface, BaseRetriever
):  # pylint: disable=too-many-instance-attributes
    """Interface for vector database technology, its connection, configs and operations"""

    db_host: str = os.environ.get("POSTGRES_DB_HOST", "localhost")
    db_port: str = os.environ.get("POSTGRES_DB_PORT", "5432")
    db_path: Optional[str] = None  # Path for a local DB, if that is being used
    collection_name: str = os.environ.get(
        "POSTGRES_DB_NAME", "adotbcollection")
    db_user = os.environ.get("POSTGRES_DB_USER", "admin")
    db_password = os.environ.get("POSTGRES_DB_PASSWORD", "secret")
    embedding: EmbeddingInterface = None
    db_client = None

    def __init__(
        self,
        embedding: EmbeddingInterface = None,
        host=None,
        port=None,
        path=None,
        collection_name=None,
        # pylint: disable=super-init-not-called
        **kwargs,
    ) -> None:  # pylint: disable=super-init-not-called
        """Instantiate a chroma client"""
        # You MUST set embedding with PGVector,
        # since with this DB type the embedding
        # dimension size always hard-coded on init
        if embedding is None:
            raise ValueError(
                "You MUST set embedding with PGVector,"
                "since with this DB type the embedding dimension "
                "size always hard-coded on init"
            )
        self.embedding = embedding
        self.labels = kwargs.get("labels", ["tyndale_open"])
        self.query_limit = kwargs.get("query_limit", QUERY_LIMIT)
        self.max_cosine_distance = kwargs.get(
            "max_cosine_distance", MAX_COSINE_DISTANCE
        )
        if host:
            self.db_host = host
        if port:
            self.db_port = port
        user = kwargs.get("user")
        password = kwargs.get("password")
        if password is not None:
            self.db_password = password
        if user is not None:
            self.db_user = user
        self.db_path = path
        if collection_name:
            self.collection_name = collection_name
        try:
            self.db_conn = psycopg2.connect(
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
                dbname=self.collection_name,
            )
            cur = self.db_conn.cursor()

            # install pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.db_conn.commit()

            # Register the vector type with psycopg2
            register_vector(self.db_conn)

            if self.embedding:
                test_doc = schema.Document(docId="test", text="test")
                self.embedding.get_embeddings([test_doc])
                embedding_vector_size = len(test_doc.embedding)
                log.info(
                    "Using PGVector embedding dimension size: %s", embedding_vector_size
                )
            else:
                embedding_vector_size = (
                    schema.EmbeddingDimensionSize.HUGGINGFACE_DEFAULT
                )
                log.info(
                    "Using PGVector embedding dimension size: %s", embedding_vector_size
                )

            # Create table to store embeddings and metadata
            table_create_command = f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                        id bigserial primary key,
                        source_id text unique,
                        document text,
                        label text,
                        media text,
                        links text,
                        embedding vector({embedding_vector_size}),
                        metadata jsonb
                        );
                        """

            cur.execute(table_create_command)
            cur.close()
            self.db_conn.commit()
        except Exception as exe:
            raise PostgresException(
                "While initializing client: " + str(exe)) from exe

    def add_to_collection(self, docs: List[schema.Document], **kwargs) -> None:
        """Loads the document object as per chroma DB formats into the collection"""
        data_list = []
        for doc in docs:
            cur = self.db_conn.cursor()
            cur.execute(
                "SELECT 1 FROM embeddings WHERE source_id = %s", (doc.docId,))
            doc_id_already_exists = cur.fetchone()
            if not doc_id_already_exists:
                data_list.append(
                    [
                        doc.docId,
                        doc.text,
                        doc.label,
                        doc.media,
                        doc.links,
                        doc.embedding,
                    ]
                )
            else:
                # Update instead of add
                cur.execute(
                    """
                    UPDATE embeddings 
                    SET 
                        document = %s, 
                        label = %s, 
                        media = %s, 
                        links = %s, 
                        embedding = %s 
                    WHERE 
                        source_id = %s
                    """,
                    (
                        doc.text,
                        doc.label,
                        doc.media,
                        doc.links,
                        doc.embedding,
                        doc.docId,
                    ),
                )
            cur.close()
        try:
            cur = self.db_conn.cursor()
            execute_values(
                cur,
                "INSERT INTO embeddings (source_id, document, label, media, links, embedding"
                ") VALUES %s",
                data_list,
            )
            self.db_conn.commit()

            # create index
            cur.execute("SELECT COUNT(*) as cnt FROM embeddings;")
            num_records = cur.fetchone()[0]
            num_lists = num_records / 1000
            num_lists = max(10, num_lists, math.sqrt(num_records))
            # use the cosine distance measure, which is what we'll later use for querying
            cur.execute(
                "CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) "
                + f"WITH (lists = {num_lists});"
            )
            self.db_conn.commit()

            cur.close()
        except Exception as exe:
            raise PostgresException("While adding data: " + str(exe)) from exe

    def get_relevant_documents(self, query: list, **kwargs) -> List[LangchainDocument]:
        """Similarity search on the vector store"""
        query_doc = schema.Document(docId="xxx", text=query)
        try:
            self.embedding.get_embeddings(doc_list=[query_doc])
            query_vector = query_doc.embedding
        except Exception as exe:
            raise GenericException(
                "While vectorising the query: " + str(exe)) from exe
        try:
            cur = self.db_conn.cursor()
            cur.execute(
                """
                SELECT source_id, document 
                FROM embeddings 
                WHERE label = ANY(%s) 
                AND embedding <=> %s < %s 
                ORDER BY embedding <=> %s 
                LIMIT %s
                """,
                (
                    self.labels,
                    np.array(query_vector),
                    self.max_cosine_distance,
                    np.array(query_vector),
                    self.query_limit,
                ),
            )
            records = cur.fetchall()
            cur.close()
        except Exception as exe:
            log.exception(exe)
            raise PostgresException(
                "While querying with embedding: " + str(exe)
            ) from exe
        if len(records) == 0:
            return [
                LangchainDocument(
                    page_content=(
                        "No relevant context documents found. "
                        "This question can't be answered, but the user could try "
                        "rewording or asking something else."
                    ),
                    metadata={
                        "source": "no records found"
                    },
                )
            ]
        return [
            LangchainDocument(page_content=doc[1], metadata={"source": doc[0]})
            for doc in records
        ]

    async def aget_relevant_documents(
        self, query: list, **kwargs
    ) -> List[LangchainDocument]:
        """Similarity search on the vector store"""
        query_doc = schema.Document(docId="xxx", text=query)
        try:
            self.embedding.get_embeddings(doc_list=[query_doc])
            query_vector = query_doc.embedding
        except Exception as exe:
            raise GenericException(
                "While vectorising the query: " + str(exe)) from exe
        try:
            cur = self.db_conn.cursor()
            cur.execute(
                """
                SELECT source_id, document 
                FROM embeddings 
                WHERE label = ANY(%s) 
                AND embedding <=> %s < %s 
                ORDER BY embedding <=> %s 
                LIMIT %s
                """,
                (
                    self.labels,
                    np.array(query_vector),
                    self.max_cosine_distance,
                    np.array(query_vector),
                    self.query_limit,
                ),
            )
            records = cur.fetchall()

            cur.close()
        except Exception as exe:
            log.exception(exe)
            raise PostgresException(
                "While querying with embedding: " + str(exe)
            ) from exe
        if len(records) == 0:
            return [
                LangchainDocument(
                    page_content=(
                        "No relevant context documents found. "
                        "This question can't be answered, but the user could try "
                        "rewording or asking something else."
                    ),
                    metadata={
                        "source": "no records found"
                    },
                )
            ]
        return [
            LangchainDocument(page_content=doc[1], metadata={"source": doc[0]})
            for doc in records
        ]

    def get_available_labels(self) -> List[str]:
        """Query DB and find out the list of labels available in metadata,
        to be used for later filtering"""
        try:
            cur = self.db_conn.cursor()
            cur.execute("SELECT distinct(label) from embeddings")
            records = cur.fetchall()
            cur.close()
        except Exception as exe:
            raise PostgresException(
                "While querying for labels: " + str(exe)) from exe
        labels = [row[0] for row in records]
        return labels
