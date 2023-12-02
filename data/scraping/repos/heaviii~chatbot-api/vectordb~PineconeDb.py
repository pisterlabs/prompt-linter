from email import generator
import os
from langchain import VectorDBQA
from llama_index import OpenAIEmbedding
from langchain.vectorstores import Pinecone

from config import Config
from services.ThreadedGenerator import ChainStreamHandler
from vectordb.BaseVector import BaseVector
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager

import pinecone 


class PineconeDb(BaseVector):

    def __init__(self, url: str, api_key: str,environment:str, index_name: str):
        self._index_name = index_name
        self._client = self.init_from_config(url, api_key, index_name)

    @classmethod
    def init_from_config(self, api_key: str, environment: str, index_name: str):
        pinecone.init(
            api_key='c23851db-0963-4e1b-b550-e61ca6f2b832',  # find at app.pinecone.io
            environment='us-west4-gcp-free'  # next to api key in console
        )
        embeddings = self.get_embedding(self)
        return Pinecone.from_existing_index(index_name, embeddings)

        
    def create_index(self, index_name):
        pinecone.create_index(index_name, dimension=1536)

    def get_index(self):
        return self._index_name

    def create_index_text(self, index_name):
           # 加载文件夹中的所有txt类型的文件
        split_docs = self.get_docs()

        # 初始化 openai 的 embeddings 对象
        embeddings = self.get_embedding(self)
        # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        
        #docsearch = Pinecone.from_documents([t.page_content for t in split_docs], embeddings, index_name=index_name)
        self._client = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)
        return True
    
    def create_index_documents(self, index_name):
        print('create_index_chromadb')
         # 加载文件夹中的所有txt类型的文件
        split_docs = self.get_docs()
        print(split_docs)

        # 初始化 openai 的 embeddings 对象
        embeddings = self.get_embedding(self)
        # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        self._client = Pinecone.from_documents([t.page_content for t in split_docs], embeddings, index_name=index_name)
        self._client.persist()
        return True
    
    def add_documents(self, split_docs):
        self._client.add_documents([t.page_content for t in split_docs])

    def add_documents(self, split_docs):
        self._client.add_texts([t.page_content for t in split_docs])
    
    @classmethod
    def get_embedding(self):
        return OpenAIEmbedding()
    
    def similarity_search(self, query: str, k: int = 1):
        return self._client.similarity_search(query=query, k=k)
    
    def retriever_search(self, query: str, k: int = 1):
        retriever = self._client.as_retriever(search_type="cosine",search_kwargs={"k":2})# search_type="mmr", k=10, alpha=0.5, beta=0.5)
        docs = retriever.get_relevant_documents(query)
        return docs
    
    def qa(self, query: str, k: int = 1):
        llm = ChatOpenAI(temperature=0, verbose=True)
        qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=self._client, return_source_documents=True)
        return qa({"query": query})
    
    def qaStream(self, query: str, k: int = 1):
        llm = ChatOpenAI(temperature=0, streaming=True, callback_manager=CallbackManager([ChainStreamHandler(generator)]), verbose=True)
        qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=self._client, return_source_documents=True)
        return qa({"query": query})

                    