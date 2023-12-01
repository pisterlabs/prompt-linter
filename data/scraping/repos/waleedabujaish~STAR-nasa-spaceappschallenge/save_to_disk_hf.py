from dotenv import load_dotenv
load_dotenv()

from llama_index import VectorStoreIndex, SimpleDirectoryReader, LangchainEmbedding, ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.storage import StorageContext
from llama_index.vector_stores import ChromaVectorStore

import chromadb

import transformers

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

chroma_client = chromadb.PersistentClient(path="distilbert")

chroma_collection = chroma_client.get_or_create_collection("distilbert")

documents = SimpleDirectoryReader('pages').load_data()

print('Pages are loaded.')

embed_model = HuggingFaceEmbedding(model_name="distilbert-base-uncased")

print('Model is loaded into GPU.')

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print('Will start indexing and embedding.')

service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    chroma_collection=chroma_collection,
    show_progress=True,
    service_context=service_context
)
