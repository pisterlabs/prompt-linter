import json
import logging
from typing import Optional, List, Tuple

import requests
from langchain.schema import Document

from .types import (
    OneOrMany,
    ID,
    IDs,
    Embedding,
    Metadata,
    validate_ids,
    maybe_cast_one_to_many,
    validate_embeddings,
    validate_metadatas,
)


logger = logging.getLogger()


class SolrCore:
    """
    Helper class to handle a Solr core
    """

    _METADATA_FIELD_PREFIX = "metadata_"
    _METADATA_FIELD_PREFIX_LENGTH = len(_METADATA_FIELD_PREFIX)

    def __init__(self, **kwargs):
        self._page_content_field = kwargs.get("page_content_field", "text_t")
        self._vector_field = kwargs.get("vector_field", "vector")
        self._core_name = kwargs.get("core_name", "langchain")
        self._url_base = kwargs.get("url_base", "http://localhost:8983/solr")

    def get_handler_url(self, handler: str):
        return f"{self._url_base}/{self._core_name}/{handler}"

    @staticmethod
    def metadata_to_solr_fields(metadata: dict) -> dict:
        solr_dict = dict()
        for key, value in metadata.items():
            try:
                field_name = SolrCore.field_name_for_metadata_key(key, type(value))
                solr_dict[field_name] = value
            except ValueError as e:
                continue

        return solr_dict

    @staticmethod
    def field_name_for_metadata_key(metadata_key: str, type_: type) -> str:
        negative_field = metadata_key[0] == "-"  # case of empty field search
        if negative_field:
            metadata_key = metadata_key[1:]

        base_name = f"{SolrCore._METADATA_FIELD_PREFIX}{metadata_key}_"
        if type_ is str:
            base_name += "s"
        elif type_ is int:
            base_name += "i"
        elif type_ is float:
            base_name += "d"
        elif type_ is bool:
            base_name += "b"
        else:
            raise ValueError(
                f"Invalid type {type_} given, only str, int, boolean and float supported"
            )

        if negative_field:
            return f"-{base_name}"

        return base_name

    @staticmethod
    def metadata_key_for_field_name(field_name: str) -> Optional[str]:
        if not field_name.startswith(SolrCore._METADATA_FIELD_PREFIX):
            return None
        work_name = field_name[
            SolrCore._METADATA_FIELD_PREFIX_LENGTH :
        ]  # remove prefix

        underscore_position = work_name.rfind("_")
        underscore_relative_position = len(work_name) - underscore_position

        if underscore_relative_position >= 3:
            raise ValueError(
                f"Unsupported solr metadata field suffix in field {field_name}"
            )

        # TODO: do we need to check suffix consistency here?
        return work_name[:underscore_position]

    def vector_search(
        self,
        vector: List[float],
        n_results: int = 4,
        where: Optional[dict[str, str]] = None,
    ) -> list:
        url = self.get_handler_url("select")

        query_params = {
            "q": "{!knn f="
            + self._vector_field
            + " topK="
            + str(n_results)
            + "}"
            + str(vector),
            "fl": "*, score",
            "output": "json",
        }

        if where:
            fq_values = []
            for field, value in where.items():
                field_name = SolrCore.field_name_for_metadata_key(field, type(value))
                if not field_name:
                    continue
                fq_values.append(f"{field_name}:{value}")

            query_params["fq"] = " AND ".join(fq_values)

        params = {"params": query_params}

        logging.debug(f"Solr search using params {json.dumps(query_params)}")

        x = requests.post(url, json=params)

        data_json = json.loads(x.text)

        results_documents = []
        results_metadatas = []
        results_distances = []
        results_embeddings = []

        for doc in data_json["response"]["docs"]:
            page_content = doc.get(self._page_content_field)
            results_documents.append(page_content)

            metadata = {}
            for solr_field, value in doc.items():
                metadata_key = SolrCore.metadata_key_for_field_name(solr_field)
                if not metadata_key or not isinstance(value, (str, int, float, bool)):
                    continue
                metadata[metadata_key] = value

            results_metadatas.append(metadata)
            results_distances.append(doc.get("score", 0))
            results_embeddings.append(doc.get(self._vector_field))

        results = {
            "documents": [results_documents],
            "metadatas": [results_metadatas],
            "distances": [results_distances],
            "embeddings": [results_embeddings],
        }

        return results

    def _validate_embedding_set(
        self,
        ids: OneOrMany[ID],
        embeddings: Optional[OneOrMany[Embedding]],
        metadatas: Optional[OneOrMany[Metadata]],
        documents: Optional[OneOrMany[Document]],
        require_embeddings_or_documents: bool = True,
    ) -> Tuple[
        IDs,
        List[Embedding],
        Optional[List[Metadata]],
        Optional[List[Document]],
    ]:
        ids = validate_ids(maybe_cast_one_to_many(ids))
        embeddings = (
            validate_embeddings(maybe_cast_one_to_many(embeddings))
            if embeddings is not None
            else None
        )
        metadatas = (
            validate_metadatas(maybe_cast_one_to_many(metadatas))
            if metadatas is not None
            else None
        )
        documents = maybe_cast_one_to_many(documents) if documents is not None else None

        # Check that one of embeddings or documents is provided
        if require_embeddings_or_documents:
            if embeddings is None and documents is None:
                raise ValueError(
                    "You must provide either embeddings or documents, or both"
                )

        # Check that, if they're provided, the lengths of the arrays match the length of ids
        if embeddings is not None and len(embeddings) != len(ids):
            raise ValueError(
                f"Number of embeddings {len(embeddings)} must match number of ids {len(ids)}"
            )
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError(
                f"Number of metadatas {len(metadatas)} must match number of ids {len(ids)}"
            )
        if documents is not None and len(documents) != len(ids):
            raise ValueError(
                f"Number of documents {len(documents)} must match number of ids {len(ids)}"
            )

        # If document embeddings are not provided, we need to compute them
        if embeddings is None and documents is not None:
            if self._embedding_function is None:
                raise ValueError(
                    "You must provide embeddings or a function to compute them"
                )
            embeddings = self._embedding_function(documents)

        if embeddings is None:
            raise ValueError(
                "Something went wrong. Embeddings should be computed at this point"
            )

        return ids, embeddings, metadatas, documents  # type: ignore

    def upsert(
        self,
        ids: OneOrMany[ID],
        embeddings: Optional[OneOrMany[Embedding]] = None,
        metadatas: Optional[OneOrMany[Metadata]] = None,
        documents: Optional[OneOrMany[Document]] = None,
    ) -> None:
        """Update the embeddings, metadatas or documents for provided ids, or create them if they don't exist.

        Args:
            ids: The ids of the embeddings to update
            embeddings: The embeddings to add. If None, embeddings will be computed based on the documents using the embedding_function set for the Collection. Optional.
            metadatas:  The metadata to associate with the embeddings. When querying, you can filter on this metadata. Optional.
            documents: The documents to associate with the embeddings. Optional.

        Returns:
            None
        """

        ids, embeddings, metadatas, documents = self._validate_embedding_set(
            ids, embeddings, metadatas, documents
        )

        solr_docs = []
        for doc_id, embedding, metadata, document in zip(
            ids, embeddings, metadatas, documents
        ):
            solr_doc = {
                "id": doc_id,
                self._vector_field: embedding,
                self._page_content_field: document,
                **SolrCore.metadata_to_solr_fields(metadata),
            }

            solr_docs.append(solr_doc)

        call_url = self.get_handler_url("update/json?commit=true")

        response = requests.post(call_url, json=solr_docs)
        logging.debug(f"Solr update response {response.text}")

    def delete(self, ids: OneOrMany[ID]) -> bool:
        ids_to_delete = maybe_cast_one_to_many(ids)
        json_body = {"delete": ids_to_delete}
        response = requests.post(
            self.get_handler_url("update?commit=true"), json=json_body
        )

        logging.debug(f"Solr delete response {response.text}")

        return response.status_code == 200

    def empty(self) -> bool:
        json_body = {"delete": {"query": "*:*"}}

        response = requests.post(
            self.get_handler_url("update?commit=true"), json=json_body
        )

        logging.debug(f"Solr delete all response {response.text}")

        return response.status_code == 200
