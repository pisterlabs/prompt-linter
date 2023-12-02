import logging
import sys
import openai
import os
from dotenv import load_dotenv

from llama_index import (
    GPTVectorStoreIndex,
    GPTSimpleKeywordTableIndex, 
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from llama_index.vector_stores import PineconeVectorStore
# from langchain.llms.openai import OpenAIChat
from langchain.chat_models import ChatOpenAI
from pathlib import Path

import requests
import pinecone
from llama_index.storage.storage_context import StorageContext

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("PINECONE_API_KEY")
environment = "northamerica-northeast1-gcp"
index_name = "openai"


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

meta_doc = SimpleDirectoryReader(input_files=[f"data/meta-20221231.md"]).load_data()


# LLM Predictor (gpt-3.5-turbo)
llm_predictor_chatgpt = LLMPredictor(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size_limit=512)

# Build meta document index


# for pinecone_title, wiki_title in zip(pinecone_titles, wiki_titles):

metadata_filters = {"company_title_document": "meta_10K"}
vector_store = PineconeVectorStore(
    index_name=index_name,
    environment=environment,
    metadata_filters=metadata_filters,
    namespace="meta" # should ideally be the Unique identifier - which might be CIK at this point
    # similarity search does not happen across namespaces
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
########################################
# Indexing is a pretty costly operation, only do once
meta_index = GPTVectorStoreIndex.from_documents(
    meta_doc, storage_context=storage_context, service_context=service_context
)
########################################
meta_doc.index_struct.index_id = "meta_doc_of_10K_filings"


response = meta_index.as_query_engine(service_context=service_context).query(
    "What is the revenue of 'Family of Apps' division of Meta in 2022?"
)

print(str(response))
print(response.get_formatted_sources())


# from llama_index.indices.composability.graph import ComposableGraph

# # set summaries for each city
# index_summaries = {}
# for wiki_title in wiki_titles:
#     # set summary text for city
#     index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"

# graph = ComposableGraph.from_indices(
#     GPTSimpleKeywordTableIndex,
#     [index for _, index in city_indices.items()], 
#     [summary for _, summary in index_summaries.items()],
#     max_keywords_per_chunk=50
# )
# custom_query_engines = {
#     graph.root_id: graph.root_index.as_query_engine(retriever_mode='simple', service_context=service_context)
# }

# query_engine = graph.as_query_engine(
#     custom_query_engines=custom_query_engines,
# )
