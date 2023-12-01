#%%

import pinecone
import os
import openai
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.storage import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings import OpenAIEmbedding
from llama_index.vector_stores import VectorStoreQuery
from llama_index.schema import NodeWithScore
from typing import Optional
from llama_index.response.notebook_utils import display_source_node
from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import Any, List
from llama_index.query_engine import RetrieverQueryEngine

load_dotenv()

pinecone_api_key = os.environ["PINECONE_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
pinecone_index = pinecone.Index("quickstart")
pinecone.list_indexes()

#%%

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

#%%

query_str = "Can you tell me about the key concepts for safety finetuning"
embed_model = OpenAIEmbedding()
query_embedding = embed_model.get_query_embedding(query_str)
query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)

query_result = vector_store.query(vector_store_query)
print(query_result)
      
#%%

nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))

print(len(nodes_with_scores))

#%%

for node in nodes_with_scores:
    display_source_node(node, source_length=1000)
    
#%%

class PineconeRetriever(BaseRetriever):
    """Retriever over a pinecone vector store."""

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

#%%

retriever = PineconeRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)
retrieved_nodes = retriever.retrieve(query_str)
for node in retrieved_nodes:
    display_source_node(node, source_length=1000)

#%%

query_engine = RetrieverQueryEngine.from_args(retriever)
response = query_engine.query(query_str)
print(str(response))

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%