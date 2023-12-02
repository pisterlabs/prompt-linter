from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PubMedSplitter(Runnable):
    def __init__(self, chunk_size:int=100, chunk_overlap:int=20):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap,
            length_function = len,
        )
        
    def invoke(self, doc:dict, config: Optional[RunnableConfig] = None):
        abstract_components = self.text_splitter.split_text(doc['abstract'])
        return [
            dict(pmid=doc['pmid'], title=doc['title'], abstract=abstract_component, keywords=doc['keywords']) 
            for abstract_component in abstract_components
        ]
