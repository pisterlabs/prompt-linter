import uuid
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import sqlalchemy
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .base import VectorStore


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

DEFAULT_COLLECTION_NAME = "langchain"


class PGVectorAsync(VectorStore):
    def __init__(
        self,
        session: AsyncSession,
        embedding_function: Embeddings,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: Dict[Any, Any] | None = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ):
        from app.models import CollectionStore, EmbeddingStore

        self.session = session
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self._distance_strategy = distance_strategy
        self.pre_delete_collection = pre_delete_collection
        self.override_relevance_score_fn = relevance_score_fn
        self.EmbeddingStore = EmbeddingStore
        self.CollectionStore = CollectionStore

    @classmethod
    async def create(
        cls,
        session: AsyncSession,
        embedding_function: Embeddings,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: Dict[Any, Any] | None = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ):
        store = cls(
            session=session,
            embedding_function=embedding_function,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            relevance_score_fn=relevance_score_fn,
        )

        await store.create_vector_extension()
        await store.create_collection()

        return store

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    async def create_vector_extension(self):
        await self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await self.session.commit()

    async def create_collection(self):
        if self.pre_delete_collection:
            await self.delete_collection()

        query = select(self.CollectionStore).where(
            self.CollectionStore.name == self.collection_name
        )

        result = await self.session.execute(query)
        collection = result.scalars().first()

        if collection:
            return

        self.session.add(
            self.CollectionStore(
                name=self.collection_name,
                cmetadata=self.collection_metadata,
            )
        )
        await self.session.commit()

    async def get_collection(self):
        query = select(self.CollectionStore).where(
            self.CollectionStore.name == self.collection_name
        )
        result = await self.session.execute(query)
        collection = result.scalars().first()
        return collection

    async def delete_collection(self):
        query = delete(self.CollectionStore).where(
            self.CollectionStore.name == self.collection_name
        )
        await self.session.execute(query)
        await self.session.commit()

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        if ids is None:
            return

        query = delete(self.EmbeddingStore).where(
            self.EmbeddingStore.custom_id.in_(ids)
        )
        await self.session.execute(query)
        await self.session.commit()

    async def delete_vectors_by_ids(self, ids: List[uuid.UUID]):
        query = delete(self.EmbeddingStore).where(self.EmbeddingStore.id.in_(ids))
        await self.session.execute(query)
        await self.session.commit()

    async def delete_vectors_by_custom_ids(self, custom_ids: List[str]):
        query = delete(self.EmbeddingStore).where(
            self.EmbeddingStore.custom_id.in_(custom_ids)
        )
        await self.session.execute(query)
        await self.session.commit()

    @classmethod
    async def __from(
        cls,
        texts: Iterable[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        session: AsyncSession,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ):
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        store = await cls.create(
            session=session,
            embedding_function=embedding,
            collection_name=collection_name,
            collection_metadata=kwargs.get("collection_metadata"),
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            relevance_score_fn=kwargs.get("relevance_score_fn"),
        )

        await store.aadd_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return store

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        collection = await self.get_collection()

        if not collection:
            raise Exception("Collection not found")

        for texts, embedding, metadata, id in zip(texts, embeddings, metadatas, ids):
            self.session.add(
                self.EmbeddingStore(
                    collection_id=collection.id,
                    embedding=embedding,
                    document=texts,
                    cmetadata=metadata,
                    custom_id=id,
                )
            )

        await self.session.commit()

        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        embeddings = await self.embedding_function.aembed_documents(list(texts))
        return await self.aadd_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    @classmethod
    async def afrom_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ):
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]

        return await cls.afrom_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ):
        session = cls.__get_session_from_kwargs(kwargs)

        embeddings = await embedding.aembed_documents(texts)

        return await cls.__from(
            texts=texts,
            embeddings=embeddings,
            embedding=embedding,
            metadatas=metadatas,
            session=session,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def __get_session_from_kwargs(cls, kwargs: Dict[Any, Any]) -> AsyncSession:
        session = kwargs.pop("session", None)
        if session is None:
            raise ValueError("A session must be provided")

        if not isinstance(session, AsyncSession):
            raise ValueError("A session must be an AsyncSession")

        return session

    @classmethod
    async def afrom_embeddings(
        cls,
        text_embeddings: List[tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ):
        session = cls.__get_session_from_kwargs(kwargs)
        texts = [text_embedding[0] for text_embedding in text_embeddings]
        embeddings = [text_embedding[1] for text_embedding in text_embeddings]

        return await cls.__from(
            texts=texts,
            embeddings=embeddings,
            embedding=embedding,
            metadatas=metadatas,
            session=session,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    async def afrom_existing_index(
        cls,
        session: AsyncSession,
        embedding: Embeddings,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ):
        store = cls(
            session=session,
            embedding_function=embedding,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
        )

        return store

    @property
    def distance_strategy(self):
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self.EmbeddingStore.embedding.l2_distance

        if self._distance_strategy == DistanceStrategy.COSINE:
            return self.EmbeddingStore.embedding.cosine_distance

        if self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self.EmbeddingStore.embedding.max_inner_product

        raise ValueError(
            f"Got unexpected distance strategy: {self._distance_strategy}."
            f"Should be one of {', '.join(DistanceStrategy.__members__.keys())}"
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[Any, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = await self.embedding_function.aembed_query(query)
        return await self.asimilarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[Any, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding = await self.embedding_function.aembed_query(query)
        return await self.asimilarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[Any, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[Any, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        collection = await self.get_collection()
        if not collection:
            raise Exception("Collection not found")

        filter_by = self.EmbeddingStore.collection_id == collection.id

        if filter is not None:
            filter_clauses: list[Any] = []
            for key, value in filter.items():
                IN = "in"
                if isinstance(value, dict) and IN in map(str.lower, value):
                    value_case_insensitive = {k.lower(): v for k, v in value.items()}
                    filter_by_metadata = (
                        self.EmbeddingStore.cmetadata[key]
                        .as_string()
                        .in_(value_case_insensitive[IN])
                    )
                    filter_clauses.append(filter_by_metadata)
                else:
                    if isinstance(value, bool):
                        filter_by_metadata = (
                            self.EmbeddingStore.cmetadata[key].as_boolean() == value
                        )
                    else:
                        filter_by_metadata = self.EmbeddingStore.cmetadata[
                            key
                        ].as_string() == str(value)

                    filter_clauses.append(filter_by_metadata)

            filter_by = sqlalchemy.and_(filter_by, *filter_clauses)

        results = await self.session.execute(
            select(
                self.EmbeddingStore, self.distance_strategy(embedding).label("distance")
            )
            .filter(filter_by)
            .order_by("distance")
            .join(
                self.CollectionStore,
                self.EmbeddingStore.collection_id == self.CollectionStore.id,
            )
            .limit(k)
        )

        results = results.all()

        docs_and_scores = [
            (
                Document(
                    page_content=result.EmbeddingStore.document,
                    metadata=result.EmbeddingStore.cmetadata
                    if result.EmbeddingStore.cmetadata
                    else {},
                ),
                result.distance if self.embedding_function is not None else None,
            )
            for result in results
        ]

        return docs_and_scores

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor

        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn

        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn

        if self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn

        raise ValueError(
            "No supported normalization function"
            f" for distance_strategy of {self._distance_strategy}."
            "Consider providing relevance_score_fn to PGVector constructor."
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> "PGVectorAsync":
        raise NotImplementedError

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError
