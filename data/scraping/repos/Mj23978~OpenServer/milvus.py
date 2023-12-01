from langchain.vectorstores.milvus import Milvus

from .base import VectorStore
from .embedding.base import BaseEmbedding


class MilvusVectorStore(VectorStore):
    def __init__(
            self,
            host: str,
            port: str,
            collection_name: str,
            embedding_model: BaseEmbedding,
            primary_field: str = "pk",
            text_field: str = "text",
            vector_field: str = "vector"
    ):
    
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.primary_field = primary_field
        self.text_field = text_field
        self.vector_field = vector_field
        
        self.client = Milvus(
          connection_args={"host": host, "port": port},          
          collection_name=collection_name,
          embedding_function=embedding_model.client,
          primary_field=primary_field,
          vector_field=vector_field,
          text_field=text_field,          
        )