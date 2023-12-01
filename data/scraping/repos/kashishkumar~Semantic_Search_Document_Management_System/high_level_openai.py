import os 
import openai 
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader

openai.api_key = "OPENAI_API_KEY"

documents =  SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query('What is this text about?')
print(response)

response = query_engine.query('Who is the author?')
print(response)


response = query_engine.query('What is the title?')
print(response)



embed_model =  LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
service_context = ServiceContext(embed_model, query_engine)