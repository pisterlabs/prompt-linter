from utils.logger import logging
from utils.exceptions import CustomException
from utils.utils import load_embedding_model
import sys
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from chromadb.config import Settings
import chromadb
from langchain.vectorstores import Chroma

CHROMA_SETTINGS = Settings(persist_directory="./",anonymized_telemetry=False)
load_dotenv(".env")
HUGGINGFACE_EMBEDDING_REPO_ID=os.environ.get("HUGGINGFACE_EMBEDDING_REPO_ID")
HUGGINGFACE_API_TOKEN=os.environ.get("HUGGINGFACE_API_TOKEN")
QDRANT_URL=os.environ.get("QDRANT_URL")
QDRANT_API_KEY=os.environ.get("QDRANT_API_KEY")

class load_to_qdrant:
    """collection name is qdrant collection name"""
    def __init__(self,collection_name:str) -> None:
        self.collection_name=collection_name
    
    def initialize_upload(self,data):
        try:
            logging.info("loading huggingface embedding model")
            embedding_model=load_embedding_model()
            logging.info("embedding model loaded successfully")
        except Exception as e:
            logging.info("facing difficulty in loading the embedding model")
            raise CustomException(sys,e)
        try:
            logging.info("connecting to qdrant server")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
            texts = text_splitter.split_documents(data)
            qdrant = Qdrant.from_documents(texts,embedding_model,url=QDRANT_URL,prefer_grpc=True,api_key=QDRANT_API_KEY,collection_name=self.collection_name)
            logging.info("data loaded successfully")
        except Exception as e:
            logging.info("error during data uploading to qdrant")
            raise CustomException(sys,e)


class load_to_chroma:

    def initialize_upload(self,data):
        try:
            logging.info("loading huggingface embedding model")
            embedding_model=load_embedding_model()
            logging.info("embedding model loaded successfully")
        except Exception as e:
            logging.info("facing difficulty in loading the embedding model")
            raise CustomException(sys,e)
        try:
            logging.info("connecting to chroma server")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
            texts = text_splitter.split_documents(data)
            chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path="./")
            db = Chroma(persist_directory="./", embedding_function=embedding_model, client_settings=CHROMA_SETTINGS, client=chroma_client)
            collection = db.get()
            db = Chroma.from_documents(texts, embedding_model, persist_directory="./", client_settings=CHROMA_SETTINGS, client=chroma_client)
            logging.info("data loaded successfully")
        except Exception as e:
            logging.info("error during data uploading to chroma")
            raise CustomException(sys,e)
