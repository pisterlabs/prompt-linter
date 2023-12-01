import os
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# This version of the Qdrant storer uses the CharacterTextSplitter to split the text into chunks.

# SPLITTING AND EMBEDDING SETTINGS
# Chunk size is the maximum size of a single chunk of text that will be sent to OpenAI for embedding.
# Using tiktoken, we perform the counting using tokens instead of characters.
chunk_size=2000
# Chunk overlap is the number of characters that will be repeated between chunks.
# It helps to maintain context between chunks.
chunk_overlap=200

print("Loading environment variables...")
load_dotenv()
openAI_key = os.environ["OPENAI_API_KEY"]
qdrant_key = os.environ["QDRANT_API_KEY"]
qdrant_host = os.environ["QDRANT_HOST"]
print("Environment variables loaded.")

print("Loading CSV data...")
loader = CSVLoader(
    file_path="./sources/temp/temporary_regulatory_data_ready.csv",
    source_column="Source",
)
print("CSV data loaded.")
# data = loader.load()

print("Running text splitter...")
documents = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
docs = text_splitter.split_documents(documents)
print("Text splitting complete.")

print("Running embeddings...")
embeddings = OpenAIEmbeddings()
print("Embeddings complete.")

print("Initializing Qdrant client...")
url = qdrant_host
api_key = qdrant_key
qdrant_collection = "raven"
qdrant_client = QdrantClient(url=url, api_key=api_key)
print("Qdrant client initialized.")

print("Deleting existing Qdrant collection if it exists...")
qdrant_client.delete_collection(collection_name=qdrant_collection)
print("Collection deleted.")

print("Creating new Qdrant collection...")
qdrant_client.create_collection(
    collection_name=qdrant_collection,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
print("New collection created.")

print("Populating Qdrant collection...")
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name=qdrant_collection,
)
print("Qdrant collection populated.")

print("Script execution completed.")
