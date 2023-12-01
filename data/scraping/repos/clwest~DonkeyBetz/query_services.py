import os
import pinecone
from dotenv import load_dotenv
from bs4 import BeautifulSoup as Soup
from langchain.document_loaders import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
import requests


load_dotenv()
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT"),
    )
index_name = "donkey-betz"
OPENAI_API_KEY = os.getenv("OPENAI_API")




