#!/usr/bin/env python

from llama_index import VectorStoreIndex, ListIndex, StorageContext, load_index_from_storage
from llama_index.indices.composability import ComposableGraph
from langchain.chat_models import ChatOpenAI
import gradio
import os
import openai

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

LLAMA_INDEX_ROOT_DIR = SCRIPT_DIR + "/llama_index"

llama_index_subdirs = []
for entry in os.walk(LLAMA_INDEX_ROOT_DIR):
    if "index_store.json" in entry[2]:
        llama_index_subdirs.append(entry[0])


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai.api_key = OPENAI_API_KEY

# ll_storage_context = StorageContext.from_defaults(persist_dir=LLAMA_INDEX_ROOT_DIR)
# ll_index = load_index_from_storage(ll_storage_context)
# ll_query_engine = ll_index.as_query_engine()

ll_indices = []
for subdir in llama_index_subdirs:
    ll_storage_context = StorageContext.from_defaults(persist_dir=subdir)
    ll_index = load_index_from_storage(ll_storage_context)
    ll_indices.append(ll_index)

ll_graph = ComposableGraph.from_indices(ListIndex, ll_indices, index_summaries=[""] * len(ll_indices))
ll_query_engine = ll_graph.as_query_engine()


def chatbot(input_text):
    response = ll_query_engine.query(input_text)

    return response.response

iface = gradio.Interface(fn=chatbot,
                         inputs=gradio.components.Textbox(lines=7, label="Enter your text"),
                         outputs="text",
                         title="FlowGPT Hackathon")

iface.launch(share=True)
