"""Wrapper around Elasticsearch vector database."""
from __future__ import annotations
import re
import uuid
import random
from typing import Any, Iterable, List

# modified langchain.schema Document
from .schema import Document

from langchain.schema import BaseRetriever

# added to langchain.schema
from .schema import short_info

import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import elasticsearch

# Action input에서 검색에 영향을 줄 수 있는 요소 제거

def remove_special_characters(s):
    print("\nbefore removing----------")
    print(s)
    # Remove special characters using regular expression
    s = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z0-9 .]+", "", s)
    print("\nafter removing----------")
    print(s)
    return s

def remove_author_publisher(s):
    keywords = ["Author ", "author ", "Publisher ", "publisher "]
    print("\nbefore removing----------")
    print(s)
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        s = re.sub(pattern, "", s)
    print("\nafter removing-----------")
    print(s)
    return s


class ElasticSearchBM25Retriever(BaseRetriever):
    """Wrapper around Elasticsearch using BM25 as a retrieval method.


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

    def __init__(self, client: Any, index_name: str):
        self.client = client
        self.index_name = index_name

    @classmethod
    def create(
        cls, elasticsearch_url: str, index_name: str, k1: float = 2.0, b: float = 0.75
    ) -> ElasticSearchBM25Retriever:
        from elasticsearch import Elasticsearch

        # Create an Elasticsearch client instance
        es = Elasticsearch(elasticsearch_url)

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
                }
            }
        }

        # Create the index with the specified settings and mappings
        es.indices.create(index=index_name, mappings=mappings, settings=settings)
        return cls(es, index_name)

    def add_texts(
        self,
        texts: Iterable[str],
        refresh_indices: bool = True,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the retriver.

        Args:
            texts: Iterable of strings to add to the retriever.
            refresh_indices: bool to refresh ElasticSearch indices

        Returns:
            List of ids from adding the texts into the retriever.
        """
        try:
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        requests = []
        ids = []
        for i, text in enumerate(texts):
            _id = str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "content": text,
                "_id": _id,
            }
            ids.append(_id)
            requests.append(request)
        bulk(self.client, requests)

        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids

    # 실제로 elastictool에서 elasticsearch가 이루어지는 부분
    def get_relevant_documents(self, query: str) -> List[Document]:
        with open("config.json") as f:
            config = json.load(f)
        n = config["elasticsearch_result_count"]
        # class Document(Serializable):
        # page_content: str
        # introduction : str
        # isbn : str
        model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        embed = model.encode(query)
        query_dict = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "author",
                                    "category",
                                    "introduction",
                                    "publisher",
                                    "title",
                                    "toc",
                                ],
                                "boost": 1,
                            }
                        }
                    ]
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": embed,
                "k": 10,
                "num_candidates": 50,
                "boost": 30,
            },
            "size": 10,
        }
        res = self.client.search(
            index=self.index_name, body=query_dict, request_timeout=1200
        )
        docs = []

        for r in res["hits"]["hits"]:
            docs.append(
                Document(
                    title=r["_source"]["title"],
                    introduction=r["_source"]["introduction"],
                    author=r["_source"]["author"],
                    publisher=r["_source"]["publisher"],
                    isbn=r["_source"]["isbn"],
                )
            )

        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

    def get_author_info(self, query: str) -> List[short_info]:
        query_dict: dict()
        query = remove_special_characters(query)
        if (
            "author" in query
            or "publisher" in query
            or "Author" in query
            or "Publisher" in query
        ):
            query = remove_author_publisher(query)

        query_dict = {
            "query": {
                "term": {
                "author.keyword": {
                    "value": query
                }
                }
            }
        }

        res = self.client.search(index=self.index_name, body=query_dict, request_timeout=1200)
        docs = []

        for r in res["hits"]["hits"]:
            docs.append(
                short_info(
                    title=r["_source"]["title"],
                    author=r["_source"]["author"],
                    publisher=r["_source"]["publisher"],
                    isbn=r["_source"]["isbn"],
                )
            )

        print("\nfrom_book--------------------------------------------debug")
        print(docs[0:2])
        print("--------------------------------------------debug\n\n")
        return docs[0:2]
    
    def get_title_info(self, query: str) -> List[short_info]:
        query_dict: dict()
        query = remove_special_characters(query)
        if (
            "author" in query
            or "publisher" in query
            or "Author" in query
            or "Publisher" in query
        ):
            query = remove_author_publisher(query)

        query_dict = {
            "query": {
                "term": {
                "title.keyword": {
                    "value": query
                }
                }
            }
        }

        res = self.client.search(index=self.index_name, body=query_dict,request_timeout=1200)
        docs = []

        for r in res["hits"]["hits"]:
            docs.append(
                short_info(
                    title=r["_source"]["title"],
                    author=r["_source"]["author"],
                    publisher=r["_source"]["publisher"],
                    isbn=r["_source"]["isbn"],
                )
            )

        print("\nfrom_book--------------------------------------------debug")
        print(docs[0:2])
        print("--------------------------------------------debug\n\n")
        return docs[0:2]
    
    
    def get_publisher_info(self, query: str) -> List[short_info]:
        query_dict: dict()
        query = remove_special_characters(query)
        query = remove_author_publisher(query)

        query_dict = {
            "query": {
                "term": {
                "publisher.keyword": {
                    "value": query
                }
                }
            }
        }

        res = self.client.search(index=self.index_name, body=query_dict, request_timeout=1200)
        docs = []

        for r in res["hits"]["hits"]:
            docs.append(
                short_info(
                    title=r["_source"]["title"],
                    author=r["_source"]["author"],
                    publisher=r["_source"]["publisher"],
                    isbn=r["_source"]["isbn"],
                )
            )

        print("\nfrom_book--------------------------------------------debug")
        print(docs[0:2])
        print("--------------------------------------------debug\n\n")
        return docs[0:2]
    # 작가정보를 바탕으로 elasticsearch가 이루어지는 부분
