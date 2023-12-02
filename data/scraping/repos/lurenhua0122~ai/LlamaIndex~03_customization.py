
import chromadb
import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index import StorageContext
from llama_index.llms import OpenAI
os.environ['OPENAI_API_KEY'] = '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'


service_context = ServiceContext.from_defaults(chunk_size=500, llm=OpenAI())
chroma_client = chromadb.PersistentClient()
chrome_collection = chroma_client.create_collection('quickstart')
vector_store = ChromaVectorStore(chroma_collection=chrome_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(
    documents=documents, service_context=service_context, storage_context=storage_context)

query_engine = index.as_query_engine(
    response_mode='tree_summarize', streaming=True)
response = query_engine.query("What did the author do?")
response.print_response_stream()
