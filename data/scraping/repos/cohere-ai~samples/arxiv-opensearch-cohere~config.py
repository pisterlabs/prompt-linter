import os

import cohere
from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv(".env")
cohere_vector_size_lookup = {"embed-english-light-v2.0": 1024}

COHERE_API_KEY = os.environ["COHERE_API_KEY"]
COHERE_MODEL = "embed-english-light-v2.0"
VECTOR_NAME = "cohere_vector"
VECTOR_SIZE = cohere_vector_size_lookup[COHERE_MODEL]
DATA_PATH = "data/arxiv_5000.csv"
DATA_COLUMNS = ["title", "abstract"]
EMBED_COLUMN = ["abstract"]


def get_opensearch_client(
    host: str = "localhost", port: str = 9200
) -> OpenSearch:
    # Create the client with SSL/TLS and hostname verification disabled.
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,  # enables gzip compression for request bodies
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    return client


def get_cohere_client(api_key: str):
    return cohere.Client(api_key)


# init your clients
co = get_cohere_client(COHERE_API_KEY)
client = get_opensearch_client()
