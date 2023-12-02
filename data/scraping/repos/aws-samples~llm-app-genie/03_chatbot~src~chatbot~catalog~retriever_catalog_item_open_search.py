""" Module that contains a class that represents a OpenSearch retriever catalog item. """
from dataclasses import dataclass
from typing import List

import boto3
from babel import Locale
from chatbot.open_search import OpenSearchIndexRetriever, get_credentials
from langchain.schema import BaseRetriever
from sagemaker.huggingface.model import HuggingFacePredictor
from sagemaker.session import Session

from .retriever_catalog_item import RetrieverCatalogItem


@dataclass
class OpenSearchRetrieverItem(RetrieverCatalogItem):
    """Class that represents a Amazon OpenSearch retriever catalog item."""

    region: str
    """ AWS Region """
    index_name: str
    """ OpenSearch Index Name """
    endpoint: str
    """ URL of OpenSearch index """
    embedding_endpoint_name: str
    """ SageMaker embedding endpoint """
    secret: str
    """ Secret that stores user and password to connect to OpenSearch index """

    def __init__(
        self,
        friendly_name,
        index_name,
        endpoint,
        embedding_endpoint_name: str,
        secret_id: str,
        region=None,
        top_k=3
    ):
        super().__init__(friendly_name)
        self.index_name = index_name
        self.region = region
        self.embedding_endpoint_name = embedding_endpoint_name
        self.secret = secret_id
        self.endpoint = endpoint
        self.top_k = top_k

    def get_instance(self) -> BaseRetriever:
        embeddings_endpoint_name = self.embedding_endpoint_name
        secret_id = self.secret
        endpoint = self.endpoint
        index_name = self.index_name
        region = self.region
        top_k = self.top_k

        secret = get_credentials(secret_id, region)
        os_http_auth = (secret["user"] or "admin", secret["password"])

        boto3_session = boto3.Session(region_name=region)
        session = Session(boto3_session)
        predictor = HuggingFacePredictor(
            endpoint_name=embeddings_endpoint_name, sagemaker_session=session
        )

        retriever = OpenSearchIndexRetriever(
            index_name, endpoint, http_auth=os_http_auth, embeddings_predictor=predictor, k=top_k
        )
        return retriever
