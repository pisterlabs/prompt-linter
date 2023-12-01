import sys
import os
import logging
import random

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

import streamlit.components.v1 as components

import streamlit as st

import openai
from llama_index.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    load_index_from_storage,
    LLMPredictor,
    ServiceContext,
    set_global_service_context,
)

from llama_index.tools.types import ToolMetadata
from llama_index.tools.query_engine import QueryEngineTool


from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import KnowledgeGraphRAGRetriever

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore


# logging.basicConfig(
#     stream=sys.stdout, level=logging.INFO, force=True,
# )

openai.api_type = "azure"
openai.api_base = st.secrets["OPENAI_API_BASE"]
openai.api_version = "2023-03-15-preview"  # azure gpt-3.5 turbo
openai.api_key = st.secrets["OPENAI_API_KEY"]

llm = AzureOpenAI(
    engine=st.secrets["DEPLOYMENT_NAME"],
    temperature=0,
    model="gpt-35-turbo",
)
llm_predictor = LLMPredictor(llm=llm)

embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment=st.secrets["EMBEDDING_DEPLOYMENT_NAME"],
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1,
)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
)

set_global_service_context(service_context)

# Graph Store

os.environ["NEBULA_USER"] = st.secrets["graphd_user"]
os.environ["NEBULA_PASSWORD"] = st.secrets["graphd_password"]
os.environ[
    "NEBULA_ADDRESS"
] = f"{st.secrets['graphd_host']}:{st.secrets['graphd_port']}"

space_name = "rag_workshop"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

# Storage Context

storage_context = StorageContext.from_defaults(
    persist_dir="./chatbot_storage_graph", graph_store=graph_store
)

# KG Index

kg_index = load_index_from_storage(
    storage_context=storage_context,
    service_context=service_context,
    max_triplets_per_chunk=10,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    verbose=True,
)

kg_index_query_engine = kg_index.as_query_engine(
    include_text=False,
    retriever_mode="keyword",
    response_mode="tree_summarize",
)

# Vector Index

storage_context_vector = StorageContext.from_defaults(
    persist_dir="./chatbot_storage_vector"
)
vector_index = load_index_from_storage(
    service_context=service_context, storage_context=storage_context_vector
)

vector_rag_query_engine = vector_index.as_query_engine()

# Graph RAG Retriever

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    service_context=service_context,
    with_nl2graphquery=True,
    llm=llm,
    verbose=True,
)

# Graph RAG Query Engine

# graph_rag_query_engine = RetrieverQueryEngine.from_args(
#     graph_rag_retriever,
#     service_context=service_context,
#     verbose=True,
# )

# # Query tools

# query_engine_tools = [
#     QueryEngineTool(
#         query_engine=graph_rag_query_engine,
#         metadata=ToolMetadata(
#             name="Guardians of the Galaxy Vol-3",
#             description="Provides info about the movie guardians of the galaxy vol 3, extracted from wikipedia.",
#         ),
#     ),
#     QueryEngineTool(
#         query_engine=vector_rag_query_engine,
#         metadata=ToolMetadata(
#             name="Data Chunks based on Semantic Search",
#             description="Provides info about the movie guardians of the galaxy vol 3, in the form of Data Chunks. "
#             "Will search large piece of text and extract the most relevant information.",
#         ),
#     ),
# ]

# Chatbot

# from llama_index.agent import ReActAgent
from llama_index.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
# chat_engine = ReActAgent.from_tools(
#     query_engine_tools, llm=llm, memory=memory, verbose=True
# )

chat_engine = kg_index.as_chat_engine(
    chat_mode="react",
    memory=memory,
    verbose=True,
)

# utils


def cypher_to_all_paths(query):
    # Find the MATCH and RETURN parts
    match_parts = re.findall(r"(MATCH .+?(?=MATCH|$))", query, re.I | re.S)
    return_part = re.search(r"RETURN .+", query).group()

    modified_matches = []
    path_ids = []

    # Go through each MATCH part
    for i, part in enumerate(match_parts):
        path_id = f"path_{i}"
        path_ids.append(path_id)

        # Replace the MATCH keyword with "MATCH path_i = "
        modified_part = part.replace("MATCH ", f"MATCH {path_id} = ")
        modified_matches.append(modified_part)

    # Join the modified MATCH parts
    matches_string = " ".join(modified_matches)

    # Construct the new RETURN part
    return_string = f"RETURN {', '.join(path_ids)};"

    # Remove the old RETURN part from matches_string
    matches_string = matches_string.replace(return_part, "")

    # Combine everything
    modified_query = f"{matches_string}\n{return_string}"

    return modified_query


def result_to_df(result):
    from typing import Dict

    import pandas as pd

    columns = result.keys()
    d: Dict[str, list] = {}
    for col_num in range(result.col_size()):
        col_name = columns[col_num]
        col_list = result.column_values(col_name)
        d[col_name] = [x.cast() for x in col_list]
    return pd.DataFrame(d)


def render_pd_item(g, item):
    from nebula3.data.DataObject import Node, PathWrapper, Relationship

    if isinstance(item, Node):
        node_id = item.get_id().cast()
        tags = item.tags()  # list of strings
        props = dict()
        for tag in tags:
            props.update(item.properties(tag))
        g.add_node(node_id, label=node_id, title=str(props))
    elif isinstance(item, Relationship):
        src_id = item.start_vertex_id().cast()
        dst_id = item.end_vertex_id().cast()
        edge_name = item.edge_name()
        props = item.properties()
        # ensure start and end vertex exist in graph
        if not src_id in g.node_ids:
            g.add_node(src_id)
        if not dst_id in g.node_ids:
            g.add_node(dst_id)
        g.add_edge(src_id, dst_id, label=edge_name, title=str(props))
    elif isinstance(item, PathWrapper):
        for node in item.nodes():
            render_pd_item(g, node)
        for edge in item.relationships():
            render_pd_item(g, edge)
    elif isinstance(item, list):
        for it in item:
            render_pd_item(g, it)


def create_pyvis_graph(result_df):
    from pyvis.network import Network

    g = Network(
        notebook=True,
        directed=True,
        cdn_resources="in_line",
        height="500px",
        width="100%",
    )
    for _, row in result_df.iterrows():
        for item in row:
            render_pd_item(g, item)
    g.repulsion(
        node_distance=100,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09,
    )
    return g


def query_nebulagraph(
    query,
    space_name=space_name,
    address=st.secrets["graphd_host"],
    port=9669,
    user=st.secrets["graphd_user"],
    password=st.secrets["graphd_password"],
):
    from nebula3.Config import SessionPoolConfig
    from nebula3.gclient.net.SessionPool import SessionPool

    config = SessionPoolConfig()
    session_pool = SessionPool(user, password, space_name, [(address, port)])
    session_pool.init(config)
    return session_pool.execute(query)


#### page

st.set_page_config(
    page_title="Graph RAG Chat Bot",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Demo: Graph RAG Chat Bot")


st.info(
    "See more about: [Graph RAG](https://www.siwei.io/graph-rag/) on how it works, KG built from [this demo](https://kg-llm-build.streamlit.app/).",
    icon="📃",
)

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me question from the knowledge in **Guardians of the Galaxy Vol. 3.**",
        }
    ]

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Thanks to https://github.com/carolinedlu/llamaindex-chat-with-streamlit-docs/blob/main/streamlit_app.py?ref=blog.streamlit.io
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking with NebulaGraph..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)

# how it works
with st.sidebar:
    st.markdown(
        """
## How it works
"""
    )
    prompt = st.text_input(label="", value="Who is Rocket?")

    if st.button("Inspect 🔎"):
        response = kg_index_query_engine.query(prompt)

        answer_GraphRAG = str(response)

        related_entities = list(
            list(response.metadata.values())[0]["kg_rel_map"].keys()
        )
        render_query = (
            f"MATCH p=(n)-[*1..2]-() \n  WHERE id(n) IN {related_entities} \nRETURN p"
        )

        st.markdown(
            f"""
> Query to NebulaGraph:

```cypher
{render_query}
```
"""
        )
        st.markdown("> The SubGraph Retrieved")
        result = query_nebulagraph(render_query)
        result_df = result_to_df(result)

        # create pyvis graph
        g = create_pyvis_graph(result_df)

        # render with random file name
        graph_html = g.generate_html(f"graph_{random.randint(0, 1000)}.html")

        components.html(graph_html, height=500, scrolling=True)

        # st.write(f"*Answer*: {answer_GraphRAG}")
