from langchain.document_loaders import UnstructuredPDFLoader
import glob
from qdrant_client import QdrantClient
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from .config import OPEN_AI_KEY, VECTOR_DB_PATH

embedding = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)
try:
    qdrant_client = QdrantClient(url=VECTOR_DB_PATH)
except Exception as e:
    print("Cannot connect to vector db")

qa = {}
