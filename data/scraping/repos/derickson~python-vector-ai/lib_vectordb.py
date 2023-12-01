import os


## for vector store
from langchain.vectorstores import ElasticVectorSearch

def setup_vectordb(hf,index_name):
    # Elasticsearch URL setup
    print(">> Prep. Elasticsearch config setup")
    endpoint = os.getenv('ES_SERVER', 'ERROR') 
    username = os.getenv('ES_USERNAME', 'ERROR') 
    password = os.getenv('ES_PASSWORD', 'ERROR')

    url = f"https://{username}:{password}@{endpoint}:443"

    return ElasticVectorSearch(embedding=hf, elasticsearch_url=url, index_name=index_name), url