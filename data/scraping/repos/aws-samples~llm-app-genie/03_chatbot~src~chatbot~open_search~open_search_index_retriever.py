"""Chain for question-answering against a vector database."""
from __future__ import annotations

import json
import logging
import sys
from typing import List, Tuple

import boto3
from chatbot.embeddings import SageMakerEndpointEmbeddings
from chatbot.helpers.logger import TECHNICAL_LOGGER_NAME
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import OpenSearchVectorSearch

logger = logging.getLogger(TECHNICAL_LOGGER_NAME)

try:
    from sagemaker.huggingface.model import HuggingFacePredictor
except json.decoder.JSONDecodeError:
    logger.error(
        "Unable to load HuggingFacePredictor. "
        + "Most likely there is a problem connecting to Amazon SageMaker. "
        + "Do you have AWS credentials configured? "
    )
    sys.exit(0)


def get_credentials(secret_id: str, region_name: str) -> str:
    """Retrieve credentials password for given username from AWS SecretsManager.

    Args:
        secret_id: AWS Secrets Manager id to retrieve.
        region_name: AWS region name.

    Returns:
        AWS Secrets Manager password.

    Example:
        ```python
        credentials = get_credentials("username", "us-east-1")
        ```
    """
    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_id)
    secrets_value = json.loads(response["SecretString"])
    return secrets_value


class OpenSearchIndexRetriever(BaseRetriever):
    """Retriever to search Amazon OpenSearch.

    Args:
        index_name: OpenSearch index name.
        domain_endpoint: OpenSearch domain endpoint.
        http_auth: Tuple containing OpenSearch user and password for authentication.
        embeddings_predictor: HuggingFacePredictor for embeddings.
        k: Number of documents to query for. Default: 3
        max_character_limit: Maximum character limit for each document. Default: 1000

    Example:
        ```python
        os_http_auth = ("admin", "password")

        embeddings_predictor = HuggingFacePredictor(
            endpoint_name=embeddings_endpoint_name, sagemaker_session=session
        )

        endpoint = "https://<OPEN_SEARCH_DCOMAIN_NAME>.<AWS_REGION>.es.amazonaws.com"
        index_name = "exampleindex"

        opensearchIndexRetriever = OpenSearchIndexRetriever(
            index_name, endpoint, http_auth=os_http_auth, embeddings_predictor=predictor
        )
        ```
    """

    max_character_limit: int

    k: int
    opensearchvectorsearch: OpenSearchVectorSearch
    """ Vector search for OpenSearch. """

    def __init__(
        self,
        index_name: str,
        domain_endpoint: str,
        http_auth: Tuple[str, str],
        embeddings_predictor: HuggingFacePredictor,
        k: int = 3,
        max_character_limit: int = 1000,
    ):
        os_domain_ep = domain_endpoint
        os_index_name = index_name
        sagemaker_endpoint_embeddings = SageMakerEndpointEmbeddings(
            embeddings_predictor=embeddings_predictor
        )

        opensearchvectorsearch = OpenSearchVectorSearch(
            index_name=os_index_name,
            embedding_function=sagemaker_endpoint_embeddings,
            opensearch_url=os_domain_ep,
            http_auth=http_auth,
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        super().__init__(
            k=k,
            max_character_limit=max_character_limit,
            opensearchvectorsearch=opensearchvectorsearch,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Run search on OpenSearch index and get top k documents.

        Args:
            query: Query string.

        Returns:
            list of documents from this OpenSearch index that relate to the query.
        """
        docs = self.opensearchvectorsearch.similarity_search(query, k=self.k)
        # limit to max character limit
        for doc in docs:
            doc.page_content = doc.page_content[
                : int(min(self.max_character_limit, len(doc.page_content)))
            ]
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """See base class."""
        return await super().aget_relevant_documents(query)
