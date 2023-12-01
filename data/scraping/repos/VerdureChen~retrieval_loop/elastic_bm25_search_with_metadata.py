"""Wrapper around Elasticsearch vector database."""

from __future__ import annotations

import uuid
from typing import Any, Iterable, List

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from tqdm import tqdm

class ElasticSearchBM25Retriever(BaseRetriever):
    """`Elasticsearch` retriever that uses `BM25`.

    To connect to an Elasticsearch instance that requires login credentials,
    including Elastic Cloud, use the Elasticsearch URL format
    https://username:password@es_host:9243. For example, to connect to Elastic
    Cloud, create the Elasticsearch URL with the required authentication details and
    pass it to the ElasticVectorSearch constructor as the named parameter
    elasticsearch_url.

    You can obtain your Elastic Cloud URL and login credentials by logging in to the
    Elastic Cloud console at https://cloud.elastic.co, selecting your deployment, and
    navigating to the "Deployments" page.

    To obtain your Elastic Cloud password for the default "elastic" user:

    1. Log in to the Elastic Cloud console at https://cloud.elastic.co
    2. Go to "Security" > "Users"
    3. Locate the "elastic" user and click "Edit"
    4. Click "Reset password"
    5. Follow the prompts to reset the password

    The format for Elastic Cloud URLs is
    https://username:password@cluster_id.region_id.gcp.cloud.es.io:9243.
    """

    client: Any
    """Elasticsearch client."""
    index_name: str
    """Name of the index to use in Elasticsearch."""

    @classmethod
    def create(
        cls, elasticsearch_url: str, index_name: str, k1: float = 2.0, b: float = 0.75, overwrite_existing_index: bool = False
    ) -> ElasticSearchBM25Retriever:
        """
        Create a ElasticSearchBM25Retriever from a list of texts.

        Args:
            elasticsearch_url: URL of the Elasticsearch instance to connect to.
            index_name: Name of the index to use in Elasticsearch.
            k1: BM25 parameter k1.
            b: BM25 parameter b.

        Returns:

        """
        from elasticsearch import Elasticsearch

        # Create an Elasticsearch client instance
        es = Elasticsearch(elasticsearch_url)
        if es.indices.exists(index=index_name):
            # print the index settings
            print(f'index already exists: {index_name}')
            if overwrite_existing_index:
                print(f'overwriting existing index: {index_name}')
                es.indices.delete(index=index_name)
            else:
                print(es.indices.get_settings(index=index_name))
                return cls(client=es, index_name=index_name)


        # Define the index settings and mappings
        settings = {
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": k1,
                    "b": b,
                }
            },
        }
        mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "similarity": "custom_bm25",  # Use the custom BM25 similarity
                },
                "metadata": {
                    "type": "object",
                    "enabled": True
                }
            }
        }

        # Create the index with the specified settings and mappings
        es.indices.create(index=index_name, mappings=mappings, settings=settings)
        return cls(client=es, index_name=index_name)

    def add_texts(
        self,
        texts: Iterable[Document],
        refresh_indices: bool = True,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the retriever.

        Args:
            texts: Iterable of Document objects to add to the retriever.
            refresh_indices: bool to refresh ElasticSearch indices

        Returns:
            List of ids from adding the texts into the retriever.
        """
        try:
            from elasticsearch.helpers import parallel_bulk
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        bulk_requests = []
        ids = []
        for doc in tqdm(texts, desc="Adding texts to ElasticSearch"):
            _id = str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "_id": _id,
            }
            ids.append(_id)
            bulk_requests.append(request)

            if len(bulk_requests) >= 1000:  # 设置阈值，每1000个文档执行一次批量索引操作
                for success, info in parallel_bulk(self.client, bulk_requests, index=self.index_name):
                    if not success:
                        # handle error
                        pass
                bulk_requests = []  # 清空批量请求列表

        if bulk_requests:
            for success, info in parallel_bulk(self.client, bulk_requests, index=self.index_name):
                if not success:
                    # handle error
                    pass

        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids

    def _get_relevant_documents(
        self, query: str, num_docs: int = 10, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        query_dict = {"query": {"match": {"content": query}}, "size": num_docs}
        res = self.client.search(index=self.index_name, body=query_dict)

        docs = []
        scores = []
        for r in res["hits"]["hits"]:
            page_content = r["_source"]["content"]
            metadata = r["_source"].get("metadata", {})
            score = r["_score"]  # BM25 score
            docs.append(Document(page_content=page_content, metadata=metadata))
            scores.append(score)
        # print(f"BM25 scores: {scores}")
        # print(f'docs: {docs}')
        return [(doc, score) for doc, score in zip(docs, scores)]

    def get_document_count(self) -> int:
        """Get the total number of documents in the index.

        Returns:
            Total number of documents in the index.
        """
        res = self.client.count(index=self.index_name)
        return res.get("count", 0)

    def get_document_by_id(self, doc_id: str) -> Document:
        """Get a document by its ID.

        Args:
            doc_id: ID of the document to retrieve.

        Returns:
            The document corresponding to the given ID.
        """
        query_dict = {"query": {"match": {"metadata.id": doc_id}}, "size": 1}
        res = self.client.search(index=self.index_name, body=query_dict)

        if res["hits"]["total"]["value"] > 0:
            doc = res["hits"]["hits"][0]["_source"]
            page_content = doc["content"]
            metadata = doc.get("metadata", {})
            return Document(page_content=page_content, metadata=metadata)

        return None
