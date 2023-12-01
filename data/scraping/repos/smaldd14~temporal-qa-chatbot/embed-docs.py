import importlib
import openai
import pinecone
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

pinecone.init(
    api_key = os.environ["PINECONE_API_KEY"],
    environment = os.environ["PINECONE_ENV"]
)

index_name = os.environ["PINECONE_INDEX_NAME"]

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    print("Index does not exist, creating it")
    # we create a new index
    pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=1536  
)
embeddings = OpenAIEmbeddings()


# Import the chunk-docs module
chunk_docs = importlib.import_module('chunk-docs')

# # Get the markdown files
markdown_files = chunk_docs.get_markdown_files()

# # Split the markdown files
split_docs = chunk_docs.split_markdown_files(markdown_files)

# split_docs is a list of lists of langchain docs, loop thru them and add them to the index
for doc in split_docs:
    vector_store = Pinecone.from_documents(doc, embeddings, index_name=index_name)

