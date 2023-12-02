from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from utils.process_info import process_info
class PineconeService():
    def __init__(self, api_key, environment):
        self.index_name = "text-chat"
        self.embeddings = OpenAIEmbeddings() 
        # initialize pinecone
        pinecone.init(
            api_key=api_key,  # find at app.pinecone.io
            environment=environment  # next to api key in console
        )

    def get_docsearch(self):
        return self.docsearch
    
    def set_database(self):
        self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

    def create_database(self, file_name):
        self.docsearch = Pinecone.from_documents(process_info(file_name), self.embeddings, index_name=self.index_name)