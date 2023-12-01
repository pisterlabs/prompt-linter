import chromadb
from chromadb.config import Settings
from llama_index.vector_stores import ChromaVectorStore
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from node__parser import nodes
from web.app import Test


db2 = chromadb.PersistentClient("src/data")
collection = db2.get_collection(name="embedding_vector")
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="ai-forever/sbert_large_nlu_ru")
)
service_context = ServiceContext.from_defaults(embed_model=embed_model)


vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context,
)
index.insert_nodes(nodes)


query_engine = index.as_query_engine()
response = query_engine.query(Test)
