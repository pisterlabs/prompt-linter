"""
Build vector index for each transcript in input_folder_path with llamaindex
Uses OpenAI GPT 3.5 to compute embeddings, needs OPENAI_API_KEY in .env file
Store it in Weaviate Cloud, at WEAVIATE_URL/WEAVIATE_API_KEY defined in .env file
"""
import os

import openai
import weaviate
from dotenv import load_dotenv
from llama_index import ServiceContext
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import WeaviateVectorStore
from rich import print

load_dotenv()

input_folder_path = "../data/transcripts"
weaviate_index = "LlamaIndex"

openai.api_key = os.environ["OPENAI_API_KEY"]

transcripts_reader = SimpleDirectoryReader(input_dir=input_folder_path)
transcripts = transcripts_reader.load_data()

service_context = ServiceContext.from_defaults(
    llm=OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="You are an expert on the Streamlit Python library. Your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts. Do not hallucinate features.",
    )
)

weaviate_client = weaviate.Client(
    url=os.environ["WEAVIATE_URL"],
    auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
    additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
)

vector_store = WeaviateVectorStore(
    weaviate_client=weaviate_client, index_name=weaviate_index
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Computing Weaviate llamaindex")
VectorStoreIndex.from_documents(
    transcripts, service_context=service_context, storage_context=storage_context
)

print("Success !")
