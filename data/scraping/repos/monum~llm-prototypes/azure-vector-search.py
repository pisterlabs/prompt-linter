import os  
import datetime
import openai  
from flask import Flask
from flask import request
from flask_cors import CORS

from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index import Document
from azure.storage.blob import BlobServiceClient

# from tenacity import retry, wait_random_exponential, stop_after_attempt  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import Vector  
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SearchField,  
    VectorSearch,  
    VectorSearchAlgorithmConfiguration,  
)  
import config

from llama_index import download_loader
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser

########################### Set up ############################
# Set up azure cognitive search
service_endpoint = config.AZURE_SEARCH_ENDPOINT
index_name = config.AZURE_SEARCH_INDEX_NAME
key = config.AZURE_SEARCH_ADMIN_KEY

# Set up OpenAI
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2023-05-15"
# os.environ["OPENAI_API_BASE"] = config.AZURE_OPENAI_ENDPOINT
# os.environ["OPENAI_API_KEY"] = config.AZURE_OPENAI_API_KEY
openai.api_key = config.OPENAI_API_KEY 

# Set up flask app
app = Flask(__name__)
CORS(app)

########################### Flask App Routes ############################
@app.route("/query", methods=["POST"])
def query():
    request_data = request.get_json()
    query = request_data['question']
    if query is None:
        return "No text found:(", 201
    print("User query: " + query)

    # Source file retrieval
    try: 
        search_client = SearchClient(service_endpoint, index_name, credential=AzureKeyCredential(key))
        results = search_client.search(  
            search_text=query,  
            select=["department", "organization", "filename", "date", "content", "url"],
        )
    except Exception as e:
        print(e)
        return "Error: failed to retrieve relevant source files", 500

    # LLM response
    # construct documents from source files
    docs = []
    sources = []
    for result in results:
        doc = Document(
            text=result["content"]
        )
        doc.extra_info = {
            "department": result["department"],
            "organization":result["organization"],
            "filename": result["filename"],
            "url": result["url"],
            "date": result["date"],
            # "content": result["content"],
            "relevance": result['@search.score']
        }
        # source = {
        #     "department": result["department"],
        #     "organization":result["organization"],
        #     "filename": result["filename"],
        #     "url": result["url"],
        #     "date": result["date"],
        #     "content": result["content"],
        #     "score": result['@search.score']
        # }
        docs.append(doc)
        # sources.append(source)
    try:
        index = VectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine()
        res = query_engine.query(query)
        response = {
            "answer": res.response,
            "confidence": "", 
            "sources": res.source_nodes
        }
    except Exception as e:
        print(e)
        return "Error: failed to generate response from source files", 500

    return response, 200

@app.route("/upload/url", methods=["POST"])
def upload_url():
    url = request.form.get("url", None)
    # Load data from url using LlamaIndex loader
    SimpleWebPageReader = download_loader("SimpleWebPageReader")
    loader = SimpleWebPageReader()
    documents = loader.load_data(urls=[url])

    node_parser = SimpleNodeParser.from_defaults(chunk_size=config.NODE_PARSER_CHUNK_SIZE, chunk_overlap=config.NODE_PARSER_CHUNK_OVERLAP)
    nodes = node_parser.get_nodes_from_documents(documents)

    # Store in Cognitive Search index
    try:
        index_docs = []
        for document in nodes:
            description = request.form.get("description", None)
            department = request.form.get("label", None)
            org = request.form.get("org", None)
            content_text = document.text
            search_index_entry = {
                "id": document.doc_id,
                "description": description,
                "content": content_text,
                "department": department,
                "organization": org,
                "filename": url,
                "url": url,
                "date": str(datetime.date.today()),
                "description_vector": generate_embeddings(description),
                "content_vector": generate_embeddings(content_text)
            }
            index_docs.append(search_index_entry)
        search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=AzureKeyCredential(key))
        result = search_client.merge_or_upload_documents(documents = index_docs)  
        print("Upload of new document succeeded: {}".format(result[0].succeeded))
    except Exception as e:
        print(e)
        return "Error: {}".format(str(e)), 500
    
    return "Url uploaded!", 200


@app.route("/upload/file", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return "Please send a POST request with a file", 400
    # Read file to local directory 
    filepath = None
    try:
        new_file = request.files["file"]
        print(new_file)
        filename = new_file.filename
        filepath = os.path.join('documents', os.path.basename(filename))
        new_file.save(filepath)

        documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=8000, chunk_overlap=200)
        nodes = node_parser.get_nodes_from_documents(documents)
    except Exception as e:
        print(e)
        if filepath is not None and os.path.exists(filepath):
            os.remove(filepath)
        return "Error: {}".format(str(e)), 500
    
    # Store in blob storage
    try:
        blob_service_client = BlobServiceClient.from_connection_string(config.AZURE_STORAGE_ACCESS_KEY)
        blob_client = blob_service_client.get_blob_client(container=config.AZURE_STORAGE_CONTAINER, blob=documents[0].extra_info['file_name'])
        print("\nUploading to Azure Storage as blob:\n\t" + documents[0].extra_info['file_name'])
        with open(file=filepath, mode="rb") as data:
            blob_client.upload_blob(data)
        url = blob_client.url
    except Exception as e: # if file already exists, ask if continue to upload
        if e.error_code == "BlobAlreadyExists":
            print("Blob already exists!")
            return "Blob already Exists!", 201
        return "Error: {}".format(str(e)), 500
    
    # Store in Cognitive Search index
    try:
        index_docs = []
        for document in nodes:
            description = request.form.get("description", None)
            file_name = document.extra_info['file_name']
            department = request.form.get("label", None)
            org = request.form.get("org", None)
            content_text = document.text
            search_index_entry = {
                "id": document.doc_id,
                "description": description,
                "content": content_text,
                "department": department,
                "organization": org,
                "filename": file_name,
                "url": url,
                "date": str(datetime.date.today()),
                "description_vector": generate_embeddings(description),
                "content_vector": generate_embeddings(content_text)
            }
            index_docs.append(search_index_entry)
        search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=AzureKeyCredential(key))
        result = search_client.merge_or_upload_documents(documents = index_docs)  
        print("Upload of new document succeeded: {}".format(result[0].succeeded))
    except Exception as e:
        print(e)
        if filepath is not None and os.path.exists(filepath):
            os.remove(filepath)
        return "Error: {}".format(str(e)), 500
    
    os.remove(filepath)
    return "File uploaded!", 200

@app.route("/get_files", methods=["GET"])
def get_files():
    try:
        # blob_service_client = BlobServiceClient.from_connection_string(config.AZURE_STORAGE_ACCESS_KEY)
        # container_client = blob_service_client.get_container_client(container=config.AZURE_STORAGE_CONTAINER) 
        # blob_list = container_client.find_blobs_by_tags("category")
        label = request.args.get("label")
        search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=AzureKeyCredential(key))
        results = search_client.search(
            search_text="*",
            filter="department eq '"+label+"'",
            select="filename, url"
        )
    except Exception as e:
        print(e)
        return "Error: {}".format(str(e)), 500
    files = []
    for result in results:
        print(result)
        files.append({"name": result["filename"], "url": result["url"]}) # get url
    return files, 200

########################### Embeddings & Indexing #########################
def generate_embeddings(text):
    '''
    Generate embeddings from text string
    input: text string to be embedded
    output: text embeddings
    '''
    response = openai.Embedding.create(input=text, engine=config.OPENAI_EMBEDDING_MODEL)
    embeddings = response['data'][0]['embedding']
    return embeddings

def create_search_index(index_client):
    '''
    Create a search index with config settings
    '''
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SearchableField(name="description", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="department", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="organization", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="filename", type=SearchFieldDataType.String, filterable=True, searchable=True),
        SearchableField(name="url", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="date", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="description_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, dimensions=1536, vector_search_configuration="my-vector-config"),
        SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, dimensions=1536, vector_search_configuration="my-vector-config"),
    ]
    vector_search = VectorSearch(
        algorithm_configurations=[
            VectorSearchAlgorithmConfiguration(
                name="my-vector-config",
                kind="hnsw",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ]
    )
    # Create the search index with the semantic settings
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    result = index_client.create_or_update_index(index)
    print(f' {result.name} created')

def search_index_exist(index_client):
    index_names = index_client.list_index_names()
    for name in index_names:
        if name == index_name:
            return True
    return False

if __name__ == "__main__":
    index_client = SearchIndexClient(
        endpoint=service_endpoint, 
        credential=AzureKeyCredential(key)
    )
    # if does not exist vector store, create one
    if (not search_index_exist(index_client)): 
        print(index_name, " does not exist, creating new search index...")
        create_search_index(index_client)

    app.run(host="0.0.0.0", port=5601)