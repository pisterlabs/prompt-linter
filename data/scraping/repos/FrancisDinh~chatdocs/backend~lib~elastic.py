import logging
from typing import Any
from langchain.schema import LLMResult

from elasticsearch import Elasticsearch

from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from lib.config import Config

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO)

class Elastic:

    def __init__(self, index_name):
        self.embeddings = OpenAIEmbeddings()
        self.index_name = index_name
        self.es_url=Config.ES_URL,
        

    def vectorSearch(self, agent_id, query):
        db = ElasticVectorSearch(
            elasticsearch_url=self.es_url,
            index_name=self.index_name,
            embedding=self.embeddings
        )

        docs = db.similarity_search(query=query, filter=dict(agent_id=agent_id), k=2)
        
        return docs
    
    def pageSearch(self, pages):
        es = Elasticsearch(hosts=Config.ES_URL)
        docs = []
        
        for page in pages:
            result = es.get(index=self.index_name, id=page.get("value"))

            docs.append(Document(
                page_content=result["_source"]["text"],
                metadata=result["_source"]["metadata"],
            ))
                

        return docs
    
