import pinecone
from langchain.vectorstores import Pinecone
import os
from dotenv import load_dotenv

class vstoreInstance:
    def __init__(self, pinecone_api_key):
        self.pinecone_api_key = pinecone_api_key
        load_dotenv()  
        self.index_name = os.getenv("INDEXNAME")  
        pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")

    def get_index(self):
        return pinecone.Index(self.index_name)

    def get_vector_store(self,index, embedder, text_field):
        return Pinecone(index, embedder.embed_query, text_field)