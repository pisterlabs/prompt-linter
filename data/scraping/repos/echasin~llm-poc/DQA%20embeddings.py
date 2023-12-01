# Issues
# 1. If you rerun the is code it will upsert new embeddings into the index. 
# 2. It creates 5880 Vectors
# 3. How can set metadata in the pinecone index to stores the filename and embedding data at the document level
# 4. How can pinecone return metadata do we can track what documents have been processed in database

# importing libraries
import os
from dotenv import load_dotenv
import openai
import pinecone
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Load the environment variables from the .env file.
load_dotenv()

# Get the value of the API_KEY environment variable.

# OpenAI credentials
api_key = os.getenv('openai_api_key')
# Pinecone credentials

api_key = os.getenv('api_key')
environment = os.getenv('environment')
index_name = os.getenv('index_name')

# Source Files
directory_path = '.\dataset'

# Load the documents
def load_docs(directory_path):
    loader = DirectoryLoader(directory_path)
    documents = loader.load()
    return documents

documents = load_docs(directory_path)
print("Total number of documents :",len(documents))

# Document Splitting for Efficient Processing with LangChain:
def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs
docs = split_docs(documents)
print("Length of documents: ", len(docs))

# Create the embeddings object
embeddings = OpenAIEmbeddings(model ="text-embedding-ada-002") # type: ignore

#  Create the vector store object
pinecone.init(
    api_key  = api_key , # type: ignore
    environment = environment) # type: ignore


print("index_name: ", index_name)

