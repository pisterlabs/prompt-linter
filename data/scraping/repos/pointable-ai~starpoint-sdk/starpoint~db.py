import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from uuid import UUID

import openai
import requests
import validators

from starpoint import reader, writer, _utils

LOGGER = logging.getLogger(__name__)


class Client(object):
    """Client that combines Reader and Writer. It is recommended that one use this client rather than
    Reader and Writer independently."""

    def __init__(
        self,
        api_key: UUID,
        reader_host: Optional[str] = None,
        writer_host: Optional[str] = None,
    ):
        self.writer = writer.Writer(api_key=api_key, host=writer_host)
        self.reader = reader.Reader(api_key=api_key, host=reader_host)

        # Consider a wrapper around openai once this class gets bloated
        self.openai = None

    def delete(
        self,
        documents: List[str],
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[Any, Any]:
        """Remove documents in an existing collection. `delete()` method from [`Writer`](#writer-objects).

        Args:
            documents: The documents to remove from the collection.
            collection_id: The collection's id to remove the documents from.
                This or the `collection_name` needs to be provided.
            collection_name: The collection's name to remove the documents from.
                This or the `collection_id` needs to be provided.

        Returns:
            dict: delete response json

        Raises:
            ValueError: If neither collection id and collection name are provided.
            ValueError: If both collection id and collection name are provided.
        """
        return self.writer.delete(
            documents=documents,
            collection_id=collection_id,
            collection_name=collection_name,
        )

    def insert(
        self,
        documents: List[Dict[Any, Any]],
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[Any, Any]:
        """Insert documents into an existing collection. `insert()` method from [`Writer`](#writer-objects).

        Args:
            documents: The documents to insert into the collection.
            collection_id: The collection's id to insert the documents to.
                This or the `collection_name` needs to be provided.
            collection_name: The collection's name to insert the documents to.
                This or the `collection_id` needs to be provided.

        Returns:
            dict: insert response json

        Raises:
            ValueError: If neither collection id and collection name are provided.
            ValueError: If both collection id and collection name are provided.
            requests.exceptions.SSLError: Failure likely due to network issues.
        """
        return self.writer.insert(
            documents=documents,
            collection_id=collection_id,
            collection_name=collection_name,
        )

    def column_insert(
        self,
        embeddings: List[Dict[str, List[float] | int]],
        document_metadatas: List[Dict[Any, Any]],
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[Any, Any]:
        """Insert documents into an existing collection by embedding and document metadata arrays.
        The arrays are zipped together and inserted as a document in the order of the two arrays.
        `column_insert()` method from [`Writer`](#writer-objects).

        Args:
            embeddings: A list of embeddings.
                Order of the embeddings should match the document_metadatas.
            document_metadatas: A list of metadata to be associated with embeddings.
                Order of these metadatas should match the embeddings.
            collection_id: The collection's id to insert the documents to.
                This or the `collection_name` needs to be provided.
            collection_name: The collection's name to insert the documents to.
                This or the `collection_id` needs to be provided.

        Returns:
            dict: insert response json

        Raises:
            ValueError: If neither collection id and collection name are provided.
            ValueError: If both collection id and collection name are provided.
            requests.exceptions.SSLError: Failure likely due to network issues.
        """
        return self.writer.column_insert(
            embeddings=embeddings,
            document_metadatas=document_metadatas,
            collection_id=collection_id,
            collection_name=collection_name,
        )

    def query(
        self,
        sql: Optional[str] = None,
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        query_embedding: Optional[List[float] | Dict[str, List[float] | int]] = None,
        params: Optional[List[Any]] = None,
        text_search_query: Optional[List[str]] = None,
        text_search_weight: Optional[float] = None,
        tokenizer_type: Optional[reader.TokenizerType] = None,
    ) -> Dict[Any, Any]:
        """Queries a collection. This could be by sql or query embeddings.
        `query()` method from [`Reader`](#reader-objects).

        Args:
            sql: Raw SQL to run against the collection.
            collection_id: The collection's id where the query will happen.
                This or the `collection_name` needs to be provided.
            collection_name: The collection's name where the query will happen.
                This or the `collection_id` needs to be provided.
            query_embedding: An embedding to query against the collection using similarity search.
                This is of the shape {"values": List[float], "dimensionality": int}
            params: values for parameterized sql

        Returns:
            dict: query response json

        Raises:
            ValueError: If neither collection id and collection name are provided.
            ValueError: If both collection id and collection name are provided.
            requests.exceptions.SSLError: Failure likely due to network issues.
        """

        return self.reader.query(
            sql=sql,
            collection_id=collection_id,
            collection_name=collection_name,
            query_embeddings=query_embedding,
            params=params,
            text_search_query=text_search_query,
            text_search_weight=text_search_weight,
            tokenizer_type=tokenizer_type,
        )

    def infer_schema(
        self,
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[Any, Any]:
        """Infers the schema of a particular collection.
        Gives the results back by column name and the inferred type for that column.
        `infer_schema()` method from [`Reader`](#reader-objects).

        Args:
            collection_id: The collection's id where the query will happen.
                This or the `collection_name` needs to be provided.
            collection_name: The collection's name where the query will happen.
                This or the `collection_id` needs to be provided.

        Returns:
            dict: infer schema response json

        Raises:
            ValueError: If neither collection id and collection name are provided.
            ValueError: If both collection id and collection name are provided.
            requests.exceptions.SSLError: Failure likely due to network issues.
        """
        return self.reader.infer_schema(
            collection_id=collection_id,
            collection_name=collection_name,
        )

    def update(
        self,
        documents: List[Dict[Any, Any]],
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[Any, Any]:
        """Update documents in an existing collection. `update()` method in
        [`Writer`](#writer-objects).

        Args:
            documents: The documents to update in the collection.
            collection_id: The collection's id where the documents will be updated.
                This or the `collection_name` needs to be provided.
            collection_name: The collection's name where the documents will be updated.
                This or the `collection_id` needs to be provided.

        Returns:
            dict: update response json

        Raises:
            ValueError: If neither collection id and collection name are provided.
            ValueError: If both collection id and collection name are provided.
            requests.exceptions.SSLError: Failure likely due to network issues.
        """
        return self.writer.update(
            documents=documents,
            collection_id=collection_id,
            collection_name=collection_name,
        )

    def column_update(
        self,
        ids: List[str],
        embeddings: List[Dict[str, List[float] | int]],
        document_metadatas: List[Dict[Any, Any]],
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[Any, Any]:
        """Updates documents for an existing collection by embedding and document metadata arrays.
        The arrays are zipped together and updates the document in the order of the two arrays.
        `column_update()` method from [`Writer`](#writer-objects).

        Args:
            embeddings: A list of embeddings.
                Order of the embeddings should match the document_metadatas.
            document_metadatas: A list of metadata to be associated with embeddings.
                Order of these metadatas should match the embeddings.
            collection_id: The collection's id where the documents will be updated.
                This or the `collection_name` needs to be provided.
            collection_name: The collection's name where the documents will be updated.
                This or the `collection_id` needs to be provided.

        Returns:
            dict: update response json

        Raises:
            ValueError: If neither collection id and collection name are provided.
            ValueError: If both collection id and collection name are provided.
            requests.exceptions.SSLError: Failure likely due to network issues.
        """
        return self.writer.column_update(
            ids=ids,
            embeddings=embeddings,
            document_metadatas=document_metadatas,
            collection_id=collection_id,
            collection_name=collection_name,
        )

    def create_collection(
        self, collection_name: str, dimensionality: int
    ) -> Dict[Any, Any]:
        """Creates a collection by name and dimensionality. Dimensionality
        should be greater than 0. `create_collection()` method from [`Writer`](#writer-objects).

        Args:
            collection_name: The name of the collection that will be created.
            dimensionality: The number of dimensions the collection will have.
                Must be an int larger than 0.

        Returns:
            dict: create collections response json

        Raises:
            ValueError: If dimensionality is 0 or less.
            requests.exceptions.SSLError: Failure likely due to network issues.
        """
        return self.writer.create_collection(
            collection_name=collection_name,
            dimensionality=dimensionality,
        )

    def delete_collection(self, collection_id: str) -> Dict[Any, Any]:
        """Deletes a collection. `delete_collection()` method from [`Writer`](#writer-objects)."""
        return self.writer.delete_collection(
            collection_id=collection_id,
        )
