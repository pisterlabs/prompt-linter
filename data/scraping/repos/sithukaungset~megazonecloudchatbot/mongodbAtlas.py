import os
import getpass

MONGODB_ATLAS_CLUSTER_URI = getpass.getpass("MongoDB Atlas Cluster URI:")

# use openAI Embeddings so we need to set up our OpenAI API Key
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")


# Create a vector search index on the cluster. In the below code, embedding is the name of the field
# that contains the embedding vector. The following definition in the JSON editor on MongoDB Atlas:

{
    "mappings":{
        "dynamics":true,
        "fields": {
            "embeddings":{
                "dimensions":1536,
                "similarity":"cosine",
                "type":"knnVector"
            }
        }
    }
}

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import TextLoader

loader = TextLoader("../../../state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

from pymaongo import MongoClient

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

db_name = "langchain"
