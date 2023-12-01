import redis
import os
from dotenv import load_dotenv
from langchain.schema import Document
from typing import List, Dict
import json
from langchain.vectorstores.redis import Redis
import sys
sys.path.append(f"{os.path.dirname(__file__)}/../")
from botcore.setup import get_openai_embeddings, load_my_env

import streamlit as st

def connect_redis():
    load_my_env()
    host = st.secrets['REDIS_HOST']
    password =  st.secrets['REDIS_PASS']
    port = st.secrets['REDIS_PORT']
    db = redis.Redis(host = host, port = port, password=password, decode_responses=True)
    return db


class RedisVectorDB:

    def __init__(self):
        load_my_env()
        self.embeddings = get_openai_embeddings()
        self.url = st.secrets['REDIS_CLOUD']
        
        self.redis = {}
        self.redis['wanted'] = Redis(redis_url = self.url, index_name = "wanted",\
                embedding_function=self.embeddings.embed_query)
        self.redis['stock'] = Redis(redis_url = self.url, index_name = "stock",\
                embedding_function=self.embeddings.embed_query)
        
        self.limit = 0.2
        print("Vector DB is ready")
    
    def json_to_doc(self, data: Dict, meta_info: Dict = None) -> Document:
        """
            data = {"title": str, "features": [], "post_id": str, ...}
        """
        feats = ", ".join([i for i in data['features']])
        txt = f"{data['title']}. {feats}"
        return Document(page_content=txt, metadata=meta_info)

    ## add
    def add_new_wanted(self, data: Dict):
        doc = self.json_to_doc(data, {"type": "wanted"})
        return self.add_doc(doc, 'wanted')

    def add_new_stock(self, data: Dict):
        doc = self.json_to_doc(data, {"type": "stock"})
        return self.add_doc(doc, 'stock')
    
    def add_doc(self, doc: Document, index_name: str):
        try:
            self.redis[index_name].add_documents([doc])
            return True
        except:
            print("An exception occurred when adding new doc")
            return False

    def add_new_doc(self, doc: Document, index_name: str):
        try:
            if self.redis[index_name] is None:
                self.redis[index_name] = Redis.from_documents([doc], self.embeddings, redis_url=self.url, index_name=index_name)
            else: 
                self.redis[index_name].add_documents([doc])
            return True
        except:
            print("An exception occurred when adding document") 
            return False
    
    ## search
    def search_stock(self, wanted_data: str):
        return self.search_doc(wanted_data, "stock")

    def search_wanted(self, stock_data: Dict):
        return self.search_doc(stock_data, 'wanted')

    def search_doc(self, data: Dict, index_name: str):
        self.add_new_stock(data)
        doc = self.json_to_doc(data, {"type": index_name})
        query = doc.page_content
        try:
            results = self.redis[index_name].similarity_search_limit_score(query, score_threshold=self.limit)
            return results
        except:
            print("Error occurred when finding documents")
            return False
