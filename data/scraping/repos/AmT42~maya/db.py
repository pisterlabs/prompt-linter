from .singleton import SingletonMeta
from chromadb.config import Settings as ChromaSettings
import chromadb 
from base64 import b64encode
from core.config import settings

from langchain.vectorstores import Chroma 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


class ChromaDBService(metaclass = SingletonMeta):
    def __init__(self, name):
        credentials = b64encode(f'admin:{settings.ADMIN}'.encode("utf-8")).decode("utf-8")
        headers = {
            "Authorization" : f"Basic {credentials}"
        }
        self.name = name
        self.client = chromadb.HttpClient(settings = ChromaSettings(allow_reset = True), 
                                          host = "server", port = 8000, headers=headers)
        
        self.chroma_db = Chroma(client = self.client, embedding_function=OpenAIEmbeddings(),
                                collection_name = self.name)
        
chroma_db_service = ChromaDBService("smart_maya")
chroma_db_service_json = ChromaDBService("smart_maya_json")

def get_chroma_db():
    return chroma_db_service.chroma_db
def get_chroma_db_json():
     return chroma_db_service_json.chroma_db

text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
