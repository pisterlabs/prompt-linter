#!/usr/bin/env python3
import logging
from os import environ
from typing import List
from urllib.parse import urlparse
from opensearchpy import Urllib3AWSV4SignerAuth, __versionstr__
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.schema.embeddings import Embeddings
from boto3 import Session

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

opensearch_url = environ['ENDPOINT']
url = urlparse(opensearch_url)
region = environ.get('AWS_REGION', 'us-east-1')
service = environ.get('SERVICE', 'es')
credentials = Session().get_credentials()
auth = Urllib3AWSV4SignerAuth(credentials, region, service)

print(f"Using opensearch-py {__versionstr__}")

fake_texts = ["foo", "bar", "baz"]

class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        return [float(1.0)] * 9 + [float(0.0)]
    
docsearch = OpenSearchVectorSearch.from_texts(
    fake_texts, 
    FakeEmbeddings(), 
    opensearch_url=opensearch_url,
    use_ssl=True,
    verify_certs=True,
    http_auth=auth,
    timeout=30
)

OpenSearchVectorSearch.add_texts(
    docsearch, fake_texts, vector_field="my_vector", text_field="custom_text"
)
