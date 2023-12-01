__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import langchain

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.storage import LocalFileStore
from langchain.embeddings import BedrockEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS, Chroma

import boto3

aws_region = 'us-east-1'

langchain.debug = True
langchain.verbose = True

persist_directory = './chromadb'

bedrock_client = boto3.client("bedrock-runtime", aws_region)


def get_embeddings():
    store = LocalFileStore('./cache')
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-g1-text-02", client=bedrock_client)
    return CacheBackedEmbeddings.from_bytes_store(
                    bedrock_embeddings, store, namespace="bedrock"
                )

def load_directory():
    dir_loader = DirectoryLoader("./data")
    return [dir_loader]

embedding = get_embeddings()
loaders = load_directory()
index = VectorstoreIndexCreator(vectorstore_cls = Chroma, 
                            embedding = embedding,
                            vectorstore_kwargs={'persist_directory': persist_directory}
        ).from_loaders(loaders)
print("loaded indexes")
