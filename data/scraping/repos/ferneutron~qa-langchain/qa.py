from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

import dotenv

dotenv.load_dotenv()

PATH_SOURCES = "/app/sources.txt"
PATH_VECTORSTORE = "/app/store"

class QAModel:
    def __init__(self):
        self.load_sources()
        self.split_document()
        self.store_vectors()
        self.init_qa_retriever() 

    def __call__(self, request: str):
        return self.retriever({
            "query": request
        })

    def load_sources(self):
        with open(PATH_SOURCES) as ps:
            sources = [source for source in ps.readlines()]
        self.sources = WebBaseLoader(sources).load()

    def split_document(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100, 
            separators=["\n\n", "\n", "(?<=\. )", " ", ""])
        self.splits = splitter.split_documents(self.sources)

    def store_vectors(self):
        self.vectorstore = Chroma.from_documents(
            documents=self.splits, 
            embedding=OpenAIEmbeddings(), 
            persist_directory=PATH_VECTORSTORE)
        
    def init_qa_retriever(self):
        self.retriever = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="map_reduce",
            retriever=self.vectorstore.as_retriever(search_type="mmr"))