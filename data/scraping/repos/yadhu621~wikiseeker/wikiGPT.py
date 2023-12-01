import os
from dotenv import load_dotenv
import logging
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, OpenAIEmbedding, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load OpenAI API key .env file
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    logging.error("OPENAI_API_KEY not found in environment variables.")
    exit(1)

# Create a ServiceContext
service_context = ServiceContext.from_defaults(
    llm= OpenAI(model='gpt-4'),
    embed_model= OpenAIEmbedding(),
    node_parser= SimpleNodeParser.from_defaults(
    text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=20)),
    chunk_size=1024
)

# Initialize VectorStoreIndex and load or create index
def construct_index(directory_path):
    try:        
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
    except:
        documents = SimpleDirectoryReader(directory_path).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist()
    return index

# Construct index from html_downloads directory
index = construct_index("html_downloads")

# Define queries and get responses from the chatbot using the context information
queries = [
    "How many quotes do you have?",
    "What is your favorite quote?"
    # Add more queries as needed
]

for query in queries:
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    logging.info(f"Query: {query}\nResponse: {response}")

