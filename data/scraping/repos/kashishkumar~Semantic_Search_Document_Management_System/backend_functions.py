from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, ServiceContext, load_index_from_storage
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
import os
import openai
openai.api_key = "OPENAI_API_KEY"

def load_data(uploaded_file):
    # Create test list using uploaded file content
    text_list = []
    documents = [Document(text=text) for text in text_list]
    parser = SimpleNodeParser().from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    return documents, nodes


def index_data(documents, nodes = None, chunk_size=None):
    if nodes is not None:
        if chunk_size is not None:
            service_context = ServiceContext.from_defaults(chunk_size=chunk_size)
            index = VectorStoreIndex(nodes, service_context=service_context, show_progress=True)
        else:
            index = VectorStoreIndex(nodes, show_progress=True)
    else:
        if chunk_size is not None:
            service_context = ServiceContext.from_defaults(chunk_size=chunk_size)
            index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
        else:
            index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index


def build_query_engine(index, streaming = False):
    query_engine = index.as_query_engine(streaming=streaming)
    return query_engine


def output_response(query_engine, query):
    response = query_engine.query(query)
    return response

def show_response(response):
    print(response) 
    
def all(uploaded_file, query):
    documents, nodes = load_data(uploaded_file)
    index = index_data(documents, nodes)
    query_engine = build_query_engine(index)
    response = output_response(query_engine, query)
    show_response(response)
    
def summarize():
    return None
