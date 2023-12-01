import azure.functions as func
import logging
import time
import json
import os

# Importing custom modules
from chunkindexmanager import ChunkIndexManager
from documentindexmanager import DocumentIndexManager
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever
from langsmith import Client

# Initializing logger object
logger = logging.getLogger('liantisapi')

# Initializing FunctionApp object
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""  # Update to your API key

client = Client()

# Defining Azure Function 'IndexDocuments'
@app.function_name("IndexDocuments")
@app.route(route="IndexDocuments")
def IndexDocuments(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function triggered by HTTP request to create search indexes in Azure Search.
    """
    try:
        # Extracting configuration information from request body
        req_body = req.get_json()
        config = {
            'AZURE_SEARCH_ADMIN_KEY': req_body.get('AZURE_SEARCH_ADMIN_KEY'),
            'AZURE_SEARCH_SERVICE_ENDPOINT': req_body.get('AZURE_SEARCH_SERVICE_ENDPOINT'),
            'AZURE_SEARCH_INDEX_NAME': req_body.get('AZURE_SEARCH_INDEX_NAME'),
            'AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME': req_body.get('AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME'),
            'AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL': req_body.get('AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL'),
            'AZURE_OPENAI_API_VERSION': req_body.get('AZURE_OPENAI_API_VERSION'),
            'AZURE_OPENAI_ENDPOINT': req_body.get('AZURE_OPENAI_ENDPOINT'),
            'AZURE_OPENAI_API_KEY': req_body.get('AZURE_OPENAI_API_KEY'),
            'BLOB_CONNECTION_STRING': req_body.get('BLOB_CONNECTION_STRING'),
            'BLOB_CONTAINER_NAME': req_body.get('BLOB_CONTAINER_NAME'),
            'AZURE_SEARCH_EMBEDDING_SKILL_ENDPOINT': req_body.get('AZURE_SEARCH_EMBEDDING_SKILL_ENDPOINT'),
            'AZURE_SEARCH_KNOWLEDGE_STORE_CONNECTION_STRING': req_body.get('AZURE_SEARCH_KNOWLEDGE_STORE_CONNECTION_STRING'),
            'AI_SERVICES_RESOURCE_NAME': req_body.get('AI_SERVICES_RESOURCE_NAME'),
            'AZURE_SEARCH_COGNITIVE_SERVICES_KEY': req_body.get('AZURE_SEARCH_COGNITIVE_SERVICES_KEY'),
        }

        # Creating search indexes in Azure Search
        tenant = 'liantis'
        prefix = f"{tenant}-{config['BLOB_CONTAINER_NAME']}"
        index_resources = create_indexes(prefix, config['BLOB_CONNECTION_STRING'], config['BLOB_CONTAINER_NAME'], config)

        # Returning success message if indexes are created successfully
        return func.HttpResponse(f"Indexes Created {index_resources}", status_code=200)
    except Exception as e:
        # Logging error and returning error message if an error occurs
        logger.error(f"Could not create  Index: {str(e)}")
        return func.HttpResponse(f"Could not create  Index. {str(e)}", status_code=500)


# Defining Azure Function 'DeleteIndexes'
@app.function_name("DeleteIndexes")
@app.route(route="DeleteIndexes")
def DeleteIndexes(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function triggered by HTTP request to delete search indexes in Azure Search.
    """
    try:
        # Extracting configuration information from request body
        req_body = req.get_json()
        config = {
            'AZURE_SEARCH_ADMIN_KEY': req_body.get('AZURE_SEARCH_ADMIN_KEY'),
            'AZURE_SEARCH_SERVICE_ENDPOINT': req_body.get('AZURE_SEARCH_SERVICE_ENDPOINT'),
            'AZURE_SEARCH_INDEX_NAME': req_body.get('AZURE_SEARCH_INDEX_NAME'),
            'AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME': req_body.get('AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME'),
            'AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL': req_body.get('AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL'),
            'AZURE_OPENAI_API_VERSION': req_body.get('AZURE_OPENAI_API_VERSION'),
            'AZURE_OPENAI_ENDPOINT': req_body.get('AZURE_OPENAI_ENDPOINT'),
            'AZURE_OPENAI_API_KEY': req_body.get('AZURE_OPENAI_API_KEY'),
            'BLOB_CONNECTION_STRING': req_body.get('BLOB_CONNECTION_STRING'),
            'BLOB_CONTAINER_NAME': req_body.get('BLOB_CONTAINER_NAME'),
            'AZURE_SEARCH_EMBEDDING_SKILL_ENDPOINT': req_body.get('AZURE_SEARCH_EMBEDDING_SKILL_ENDPOINT'),
            'AZURE_SEARCH_KNOWLEDGE_STORE_CONNECTION_STRING': req_body.get('AZURE_SEARCH_KNOWLEDGE_STORE_CONNECTION_STRING')
        }

        # Deleting search indexes in Azure Search
        tenant = 'liantis'
        prefix = f"{tenant}-{config['BLOB_CONTAINER_NAME']}"
        delete_indexes(prefix, config)

        # Returning success message if indexes are deleted successfully
        return func.HttpResponse("Indexes Created", status_code=200)
    except Exception as e:
        # Logging error and returning error message if an error occurs
        logger.error(f"Could not delete Azure Search Index: {str(e)}")
        return func.HttpResponse(f"Could not delete Azure Search Index. {str(e)}", status_code=500)


@app.function_name("AskYourDocuments")
@app.route(route="AskYourDocuments")
def AskYourDocuments(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function that takes an HTTP request and returns an HTTP response.

    Args:
        req (func.HttpRequest): The HTTP request.

    Returns:
        func.HttpResponse: The HTTP response.
    """
    try:
        # Extracting configuration information from request body
        req_body = req.get_json()
        config = {
            'AZURE_SEARCH_ADMIN_KEY': req_body.get('AZURE_SEARCH_ADMIN_KEY'),
            'AZURE_SEARCH_SERVICE_ENDPOINT': req_body.get('AZURE_SEARCH_SERVICE_ENDPOINT'),
            'AZURE_SEARCH_SERVICE_NAME': req_body.get('AZURE_SEARCH_SERVICE_NAME'),
            'AZURE_SEARCH_VECTOR_INDEX_NAME': req_body.get('AZURE_SEARCH_INDEX_NAME'),
            'AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME': req_body.get('AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME'),
            'AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL': req_body.get('AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL'),
            'AZURE_OPENAI_API_VERSION': req_body.get('AZURE_OPENAI_API_VERSION'),
            'AZURE_OPENAI_ENDPOINT': req_body.get('AZURE_OPENAI_ENDPOINT'),
            'AZURE_OPENAI_API_KEY': req_body.get('AZURE_OPENAI_API_KEY'),
            'OPENAI_API_TYPE': req_body.get('OPENAI_API_TYPE'),
            'OPENAI_DEPLOYMENT_NAME': req_body.get('OPENAI_DEPLOYMENT_NAME'),
            'OPENAI_MODEL_NAME': req_body.get('OPENAI_MODEL_NAME'),
            'NUMBER_OF_CHUNKS_TO_RETURN': req_body.get('NUMBER_OF_CHUNKS_TO_RETURN'),
            'question': req_body.get('question')
        }

        # Creating instance of AzureChatOpenAI
        llm = AzureChatOpenAI(
            openai_api_base=config['AZURE_OPENAI_ENDPOINT'],
            openai_api_version=config['AZURE_OPENAI_API_VERSION'],
            deployment_name=config['OPENAI_DEPLOYMENT_NAME'],
            openai_api_key=config['AZURE_OPENAI_API_KEY'],
            openai_api_type=config['OPENAI_API_TYPE'],
            model_name=config['OPENAI_MODEL_NAME'],
            temperature=0)

        # Creating instance of AzureCognitiveSearchRetriever
        retriever = AzureCognitiveSearchRetriever(
            service_name=config['AZURE_SEARCH_SERVICE_NAME'],
            api_key=config['AZURE_SEARCH_ADMIN_KEY'],
            index_name=config['AZURE_SEARCH_VECTOR_INDEX_NAME'],
            content_key="text",
            top_k=int(config['NUMBER_OF_CHUNKS_TO_RETURN']))

        # Creating instance of RetrievalQA
        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            return_source_documents=True)

        # Generating response to user's query
        response = chain({"query": config['question']})
        source_documents = []
        for doc in response["source_documents"]:
            metadata = {
                "page_content": doc.page_content,
                "title": doc.metadata['title'],
                "score": doc.metadata['@search.score']
                # Add more metadata fields here as needed
            }
            source_documents.append(metadata)

        return func.HttpResponse(json.dumps({
                "result": response["result"],
                "source_documents": source_documents}),
            status_code=200)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise func.HttpResponse(f"Error occurred {str(e)}", status_code=500)


# Function to create search indexes in Azure Search
def create_indexes(prefix, customer_storage_connection_string, container_name, config):
    """
    Function to create search indexes in Azure Search.
    """
    index_manager = DocumentIndexManager()
    doc_index_resources = index_manager.create_document_index_resources(prefix, customer_storage_connection_string, container_name, config)
    time.sleep(5)
    chunk_index_manager = ChunkIndexManager()
    chunk_index_resources = chunk_index_manager.create_chunk_index_resources(prefix, config)  # doesnt need config
    return {"doc_index_resources": doc_index_resources, "chunk_index_resources": chunk_index_resources}


# Function to delete search indexes in Azure Search
def delete_indexes(prefix, config):
    """
    Function to delete search indexes in Azure Search.
    """
    index_manager = DocumentIndexManager()
    index_manager.delete_document_index_resources(prefix, config)
    chunk_index_manager = ChunkIndexManager()
    chunk_index_manager.delete_chunk_index_resources(prefix)
