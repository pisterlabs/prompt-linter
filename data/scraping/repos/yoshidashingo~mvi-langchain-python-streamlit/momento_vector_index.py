from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar, cast
from uuid import uuid4

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from momento import PreviewVectorIndexClient
from momento.auth import CredentialProvider
from momento.config import VectorIndexConfiguration
from momento.requests.vector_index import Item
from momento.responses.vector_index import (
    AddItemBatch,
    CreateIndex,
    DeleteItemBatch,
    Search,
)

VST = TypeVar("VST", bound="VectorStore")


class MomentoVectorIndex(VectorStore):
    """Vector Store implementation backed by Momento Vector Index.

    Momento Vector Index is a serverless vector index that can be used to store and search vectors.
    """

    _client: PreviewVectorIndexClient
    index_name: str
    text_field: str
    fields: set[str]

    def __init__(
        self,
        embedding_function: Embeddings,
        configuration: VectorIndexConfiguration,
        credential_provider: CredentialProvider,
        index_name: str = "default",
        text_field: str = "text",
        source_field: str = "source",
        **kwargs,
    ):
        """Initialize a Vector Store backed by Momento Vector Index.

        Args:
            embedding_function (Embeddings): The embedding function to use.
            configuration (VectorIndexConfiguration): The configuration to initialize the Vector Index with.
            credential_provider (CredentialProvider): The credential provider to authenticate the Vector Index with.
            index_name (str, optional): The name of the index to store the documents in. Defaults to "default".
            text_field (str, optional): The name of the metadata field to store the original text in. Defaults to "text".
            source_field (str, optional): The name of the metadata field to store the source in. Defaults to "source".
        """
        self._client = PreviewVectorIndexClient(
            configuration=configuration,
            credential_provider=credential_provider,
        )
        self.embedding_func = embedding_function
        self.index_name = index_name
        self.text_field = text_field
        # TODO refactor metadata tracking once fetching all metadata is supported
        self.fields = {source_field}

    def _create_index_if_not_exists(self, num_dimensions: int) -> bool:
        """Create index if it does not exist."""
        response = self._client.create_index(self.index_name, num_dimensions)
        if isinstance(response, CreateIndex.Success):
            return True
        elif isinstance(response, CreateIndex.IndexAlreadyExists):
            return False
        elif isinstance(response, CreateIndex.Error):
            raise response.inner_exception
        else:
            raise Exception(f"Unexpected response: {response}")

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings to add to the vectorstore.
            metadatas (Optional[List[dict]]): Optional list of metadatas associated with the texts.
            kwargs (Any): Other optional parameters. Specifically:
            - ids (List[str], optional): List of ids to use for the texts. Defaults to None, in which
                case uuids are generated.

        Returns:
            List[str]: List of ids from adding the texts into the vectorstore.
        """
        texts = list(texts)
        if metadatas is not None:
            for metadata, text in zip(metadatas, texts, strict=True):
                metadata.pop(self.text_field, None)
                self.fields.update(metadata.keys())
                metadata[self.text_field] = text
        else:
            metadatas = [{self.text_field: text} for text in texts]

        try:
            embeddings = self.embedding_func.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self.embedding_func.embed_query(x) for x in texts]

        new_index_created = self._create_index_if_not_exists(len(embeddings[0]))

        if "ids" in kwargs:
            ids = kwargs["ids"]
            if len(ids) != len(embeddings):
                raise ValueError("Number of ids must match number of texts")
        else:
            ids = [str(uuid4()) for _ in range(len(embeddings))]

        batch_size = 128
        for i in range(0, len(embeddings), batch_size):
            start = i
            end = min(i + batch_size, len(embeddings))
            items = [
                Item(id=id, vector=vector, metadata=metadata)
                for id, vector, metadata in zip(
                    ids[start:end], embeddings[start:end], metadatas[start:end], strict=True
                )
            ]

            if not new_index_created:
                # Peform a deduplicated insert
                delete_response = self._client.delete_item_batch(self.index_name, ids[start:end])
                if not isinstance(delete_response, DeleteItemBatch.Success):
                    raise ValueError(delete_response)
            response = self._client.add_item_batch(self.index_name, items)
            if isinstance(response, AddItemBatch.Success):
                pass
            elif isinstance(response, AddItemBatch.Error):
                raise response.inner_exception
            else:
                raise Exception(f"Unexpected response: {response}")

        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID.

        Args:
            ids (List[str]): List of ids to delete.
            kwargs (Any): Other optional parameters (unused)

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            return True
        response = self._client.delete_item_batch(self.index_name, ids)
        return isinstance(response, DeleteItemBatch.Success)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        res = self.similarity_search_with_score(query=query, k=k, **kwargs)
        return [doc for doc, _ in res]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents to the query string.

        Args:
            query (str): The query string to search for.
            k (int, optional): The number of results to return. Defaults to 4.
            kwargs (Any): Vector Store specific search parameters. The following are forwarded to the Momento Vector Index:
            - top_k (int, optional): The number of results to return.
            - metadata_fields (List[str], optional): The metadata fields to return.

        Returns:
            List[Tuple[Document, float]]: A list of tuples of the form (Document, score).
        """
        embedding = self.embedding_func.embed_query(query)

        results = self.similarity_search_with_score_by_vector(embedding=embedding, k=k, **kwargs)
        return results

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents to the query vector.

        Args:
            embedding (List[float]): The query vector to search for.
            k (int, optional): The number of results to return. Defaults to 4.
            kwargs (Any): Vector Store specific search parameters. The following are forwarded to the Momento Vector Index:
            - top_k (int, optional): The number of results to return.
            - metadata_fields (List[str], optional): The metadata fields to return.

        Returns:
            List[Tuple[Document, float]]: A list of tuples of the form (Document, score).
        """
        output_fields = [self.text_field] + list(self.fields)
        if "top_k" in kwargs:
            k = kwargs["k"]
        if "metadata_fields" in kwargs:
            output_fields = kwargs["metadata_fields"]
        response = self._client.search(self.index_name, embedding, top_k=k, metadata_fields=output_fields)

        if not isinstance(response, Search.Success):
            return []

        results = []
        for hit in response.hits:
            metadata = {x: hit.metadata[x] for x in output_fields if x in hit.metadata}
            text = cast(str, metadata.pop(self.text_field))
            doc = Document(page_content=text, metadata=metadata)
            pair = (doc, hit.distance)
            results.append(pair)

        return results

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs: Any) -> List[Document]:
        results = self.similarity_search_with_score_by_vector(embedding=embedding, k=k, **kwargs)
        return [doc for doc, _ in results]

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """Return the Vector Store initialized from texts and embeddings.

        Args:
            cls (Type[VST]): The Vector Store class to use to initialize the Vector Store.
            texts (List[str]): The texts to initialize the Vector Store with.
            embedding (Embeddings): The embedding function to use.
            metadatas (Optional[List[dict]], optional): The metadata associated with the texts. Defaults to None.
            kwargs (Any): Vector Store specific parameters. The following are forwarded to the Vector Store constructor
                and required:
            - configuration (VectorIndexConfiguration): The configuration to initialize the Vector Index with.
            - credential_provider (CredentialProvider): The credential provider to authenticate the Vector Index with.
            - index_name (str, optional): The name of the index to store the documents in. Defaults to "default".
            - text_field (str, optional): The name of the metadata field to store the original text in. Defaults to "text".

        Returns:
            VST: Momento Vector Index vector store initialized from texts and embeddings.
        """
        vector_db = cls(embedding_function=embedding, **kwargs)  # type: ignore
        vector_db.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return vector_db
