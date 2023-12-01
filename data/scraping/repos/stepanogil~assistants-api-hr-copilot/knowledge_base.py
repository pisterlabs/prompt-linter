from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
import os  
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI embedding model
EMBEDDING_MODEL_NAME  = os.getenv("OPENAI_API_EMBEDDING_MODEL")
EMEBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv("OPENAI_API_EMBEDDING_DEPLOYMENT")
SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")

# retrieval algorithm
RAG_RETRIEVAL_ALGORITHM = os.getenv("RAG_RETRIEVAL_ALGORITHM")

# Azure Cognitive Search vector db
vector_store_address = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
vector_store_password = os.getenv("AZURE_SEARCH_ADMIN_KEY") 
index_name = SEARCH_INDEX_NAME

# Azure embedding model
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME,
                                   deployment=EMEBEDDING_MODEL_DEPLOYMENT_NAME)

# initialize vector store
vector_store =  AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embedding_model.embed_query)

def retrieve_relevant_kb(query):
    """
    Retrieves relevant knowledge base chunks from Azure Cognitive Search.
    """    
    docs = vector_store.similarity_search(
        query=query,
        k=3,
        search_type=RAG_RETRIEVAL_ALGORITHM,
    )
    return {"doc0": docs[0].page_content, "doc1": docs[1].page_content, "doc2": docs[2].page_content}


