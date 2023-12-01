import os
from dotenv import load_dotenv
os.environ['openai_api_key'] = os.getenv('API_KEY')
from langchain.chat_models import ChatOpenAI
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, GPTEmptyIndex, StorageContext, LLMPredictor
class vector:
    def __init__(self):
        self.index = None
        self.query_engine = None
        self.documents = None
        self.query = None
        self.response = ""
        self.storage_context = None
        self.indexed = False
        self.llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"))
        self.servicecontext = ServiceContext.from_defaults(llm_predictor=self.llm_predictor)
    def onetime(self):
        self.setdocs('data')
        self.setindex()
    def init(self):
        self.loadindex()
        self.setqueryengine()
        self.indexed = True
    def isset(self):
        return self.indexed
    def loadindex(self):
        #self.storage_context = storagecontext.from_defaults(persist_dir="./files/index")
        #index with no docs
        self.index = GPTEmptyIndex(self.storage_context)
    def setdocs(self, location):
        self.documents = SimpleDirectoryReader(location).load_data()
    def setindex(self):
        self.index = VectorStoreIndex.from_documents(self.documents)
        self.index.storage_context.persist(persist_dir="./files/index")
    def setqueryengine(self):
        self.query_engine = self.index.as_chat_engine(servicecontext = self.servicecontext, verbose=True)
    def setquery(self, query):
        self.query = query
    def setresponse(self):
        self.response = self.query_engine.chat(self.query)
        return self.response
    def getresponse(self):
        return self.response
    def getquery(self):
        return self.query
    def getdocs(self):
        return self.documents
    def getindex(self):
        return self.index
