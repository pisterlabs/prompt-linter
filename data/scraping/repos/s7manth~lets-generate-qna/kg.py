"""
RAG using Knowledge Graphs. 
"""

import os
import logging
import sys

logging.basicConfig(
  stream=sys.stdout, level=logging.INFO
)

from dotenv import load_dotenv
load_dotenv()

from llama_index import (
  VectorStoreIndex,
  SimpleDirectoryReader,
  KnowledgeGraphIndex,
  ServiceContext,
)

from llama_index import set_global_service_context
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from llama_index.query_engine import KnowledgeGraphQueryEngine

import logging
import sys

from llama_index.llms import OpenAI

llm = OpenAI(temperature=0, model="gpt-4-1106-preview")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

set_global_service_context(service_context)

# connection_string = f"--address {os.environ['GRAPHD_HOST']} --port 9669 --user root --password {os.environ['NEBULA_PASSWORD']}"

space_name = "fyp"
edge_types, rel_prop_names = ["relationship"], ["relationship"]
tags = ["entity"]

graph_store = NebulaGraphStore(
  space_name=space_name,
  edge_types=edge_types,
  rel_prop_names=rel_prop_names,
  tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

documents = SimpleDirectoryReader("./data").load_data()

kg_index = KnowledgeGraphIndex.from_documents(
  documents,
  storage_context=storage_context,
  service_context=service_context,
  max_triplets_per_chunk=10,
  space_name=space_name,
  edge_types=edge_types,
  rel_prop_names=rel_prop_names,
  tags=tags,
)

nl2kg_query_engine = KnowledgeGraphQueryEngine(
  storage_context=storage_context,
  service_context=service_context,
  llm=llm,
)
