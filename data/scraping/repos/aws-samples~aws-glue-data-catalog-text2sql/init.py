import boto3
import config
import langchain
import sys
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.docstore.document import Document

if __name__ == "__main__":
    
    
    session = boto3.session.Session()
    region = config._global['region']
    credentials = session.get_credentials()
    service = 'es'
    http_auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token)
    opensearch_cluster_domain_endpoint = config.opensearch['domain_endpoint']
    domain_name = config.opensearch['domain_name']
    index_name = "index-superglue"
    
    # Create AWS Glue client
    
    glue_client = boto3.client('glue', region_name=region)
    
    
    # Create Amazon Opensearch client
    
    def get_opensearch_cluster_client():
        opensearch_client = OpenSearch(
            hosts=opensearch_cluster_domain_endpoint,
            http_auth=http_auth,
            engine="faiss",
            index_name=index_name,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )
        return opensearch_client
    
    # Function to get all tables from Glue Data Catalog
    
    
    def get_tables(glue_client):
        # get all AWS Glue databases
        databases = glue_client.get_databases()
    
        tables = []
    
        num_db = len(databases['DatabaseList'])
    
        for db in databases['DatabaseList']:
            tables = tables + \
                glue_client.get_tables(DatabaseName=db['Name'])["TableList"]
    
        num_tables = len(tables)
    
        return tables, num_db, num_tables
    
    # Function to flatten JSON representations of Glue tables
    
    
    def dict_to_multiline_string(d):
    
        lines = []
        db_name = d['DatabaseName']
        table_name = d['Name']
        columns = [c['Name'] for c in d['StorageDescriptor']['Columns']]
    
        line = f"{db_name}.{table_name} ({', '.join(columns)})"
        lines.append(line)
    
        return "\n".join(lines)
    
    # Amazon Bedrock LangChain clients
    
    
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    
    # VectorDB type
    
    vectorDB = sys.argv[1]
    
    if vectorDB == "faiss":
    
        print("INFO: Indexing FAISS started.")
    
        catalog, num_db, num_tables = get_tables(glue_client)
    
        docs = [
            Document(
                page_content=dict_to_multiline_string(x),
                metadata={
                    "source": "local"}) for x in catalog]
    
        vectorstore_faiss = FAISS.from_documents(
            docs,
            bedrock_embeddings,
        )
    
        print("INFO: Loaded Documents in FAISS.")
    
        vectorstore_faiss.save_local("faiss_index")
    
        print("COMPLETE: FAISS Index saved.")
    
    elif vectorDB == "opensearch":
    
        print("INFO: Opensearch Index saved.")
    
        catalog, num_db, num_tables = get_tables(glue_client)
    
        # Initialize Opensearch clients
    
        opensearch_client = get_opensearch_cluster_client()
    
        vectorstore_opensearch = OpenSearchVectorSearch(
            index_name=index_name,
            embedding_function=bedrock_embeddings,
            opensearch_url=opensearch_cluster_domain_endpoint,
            engine="faiss",
            timeout=300,
            use_ssl=True,
            verify_certs=True,
            http_auth=http_auth,
            connection_class=RequestsHttpConnection
        )
    
        # Delete index for initial batch embedding
    
        try:
            opensearch_client.indices.delete(index_name)
        except BaseException:
            print("Index does not exist.")
    
        # Prepare and add documents
    
        docs = [
            Document(
                page_content=dict_to_multiline_string(x),
                metadata={
                    "source": "local"}) for x in catalog]
    
        vectorstore_opensearch.add_documents(docs)
    
        print("COMPLETE: Loaded Document Embeddings in Opensearch.")
    
    
    else:
        print("ERROR: Invalid vector database type.")
