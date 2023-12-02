from langchain.vectorstores import Pinecone
import pinecone
from pinecone.index import Index
from doc_query.openai_utils import embedding_model
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

index = Index(PINECONE_INDEX)
vectorstore = Pinecone(index=index, embedding_function=embedding_model.embed_query, text_key="text")
