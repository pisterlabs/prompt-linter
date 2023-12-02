from pydantic import BaseModel
from typing import List, Optional
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
import os
import faiss
from langchain.vectorstores.utils import DistanceStrategy

PARSE_FOLDER_PATH = os.path.join(os.getcwd(), 'parse_res')
load_dotenv()
openai_api_key= os.getenv('LITELLM_OPENAI_API_KEY')
openai_api_base= os.getenv('LITELLM_OPENAI_API_BASE')
# class VectorItem(BaseModel):
#     embeddings: List[List[float]]
#     keys: List[str]
#     content: str


# class ChainItem(BaseModel):
#     embedding:List[float]
#     text: str
    

#     @property
#     def is_key(self):
#         return self.text.startswith('chainkey_')
    
#     @property
#     def key(self):
#         return self.text.replace('chainkey_',"")

#     @property
#     def content(self) -> str:
#         if self.is_key():
#             return self.key()
#         else:
#             return self.text


class IndexNode(BaseModel):
    id: str
    embedding:Optional[List[float]] = None
    content: str

class TextChunk(BaseModel):
    id: str
    embedding:Optional[List[float]] = None
    content: str
    index_node_ids: List[str]


class IndexStore():
    filter_key_words = [
        '成交量',"投资者", "投资风险", "上市公司股东","换手率", "成交额", "振幅"
    ]
    def __init__(self,):
        self.node_to_chunk = {}
        self.chunk_to_node = {}
        self.chunks:List[TextChunk] = []
        self.index_nodes:List[IndexNode] = []
        self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key,
            openai_api_base=openai_api_base)
        return
    
    def find_node(self, node_id:str) -> IndexNode:
        for node in self.index_nodes:
            if node.id == node_id: return node
        return None
    
    def add_chunk(self,chunk: TextChunk):
        if chunk not in self.chunks:
            self.chunks.append(chunk)
        for id in chunk.index_node_ids:
            node = self.find_node(node_id=id)
            if node is None:
                return
            if node.content in self.node_to_chunk.keys():
                val = self.node_to_chunk[node.content].copy()
                val.append(chunk.id)
                self.node_to_chunk[node.content] = val
            else:
                self.node_to_chunk[node.content] = [chunk.id]

    def add_node(self, index_node:IndexNode):
        if index_node not in self.index_nodes:
            self.index_nodes.append(index_node)

    def contains(self, node_content: str):
        return node_content in self.node_to_chunk.keys()

    def save(self, path):
        json_to_write = {
            "chunks":[item.dict() for item in self.chunks],
            "index_nodes": [item.dict() for item in self.index_nodes],
            "node_to_chunk": self.node_to_chunk,
            "chunk_to_node": self.chunk_to_node
        }
        json.dump(
            json_to_write,
            open(path, 'w'),
            indent=4,
            ensure_ascii=False)
    
    def load_from_file(self, path):
        index_store = json.loads(
            open(path, 'r').read()

        )
        self.node_to_chunk = index_store.get('node_to_chunk')
        self.chunk_to_node = index_store.get('chunk_to_node')
        self.chunks = [TextChunk.parse_obj(chunk) for chunk in index_store.get('chunks', [])]
        self.index_nodes = [IndexNode.parse_obj(node) for node in index_store.get('index_nodes', [])]
        self.init_vectodstore()
    
    def add_node_embeddings(self):
        texts = [item.content for item in self.index_nodes]
        embeddings = self.embedding.embed_documents(texts)
        for index,item in enumerate(self.index_nodes):
            item.embedding=embeddings[index]

    def init_vectodstore(self):
        print(len(self.index_nodes),len(self.chunks))
        node_vector_base= FAISS.from_texts(
            texts = [item.content for item in self.index_nodes],
            embedding=self.embedding,
            metadatas=[{'node_id':item.id} for item in self.index_nodes],
            distance_strategy=DistanceStrategy.COSINE
        )

        self.node_vector_store = node_vector_base
    

    def search_nodes(self,query:str)-> List[IndexNode]:
        search_result = self.node_vector_store.similarity_search_with_score(
            query=query,
            threshold=0.5
        )
        print(search_result)
        index_nodes = [self.find_node(node_id=doc.metadata['node_id']) for doc,score in search_result]
        return index_nodes




# class IndexCollections(BaseModel):
#     items: List[ChainItem]
#     id: str

