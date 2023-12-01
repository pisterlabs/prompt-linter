import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

import streamlit as st
import streamlit.components.v1 as components

import re

import random

CODE_KG_RAG = """

# Build Knowledge Graph with KnowledgeGraphIndex 

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

# Create a Graph RAG Query Engine

kg_rag_query_engine = kg_index.as_query_engine(
    include_text=False,
    retriever_mode="keyword",
    response_mode="tree_summarize",
)

"""


import os
import json
import openai
from llama_index.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
)

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_type = "azure"
openai.api_base = st.secrets["OPENAI_API_BASE"]
# openai.api_version = "2022-12-01" azure gpt-3
openai.api_version = "2023-05-15"  # azure gpt-3.5 turbo
openai.api_key = st.secrets["OPENAI_API_KEY"]

llm = AzureOpenAI(
    engine=st.secrets["DEPLOYMENT_NAME"],
    temperature=0,
    model="gpt-35-turbo",
)
llm_predictor = LLMPredictor(llm=llm)

# You need to deploy your own embedding model as well as your own chat completion model
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
os.environ["NEBULA_USER"] = st.secrets["graphd_user"]
os.environ["NEBULA_PASSWORD"] = st.secrets["graphd_password"]
os.environ[
    "NEBULA_ADDRESS"
] = f"{st.secrets['graphd_host']}:{st.secrets['graphd_port']}"

space_name = "guardians"
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

from llama_index import load_index_from_storage

storage_context = StorageContext.from_defaults(
    persist_dir="./storage_graph", graph_store=graph_store
)
kg_index = load_index_from_storage(
    storage_context=storage_context,
    service_context=service_context,
    max_triplets_per_chunk=10,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

storage_context_vector = StorageContext.from_defaults(persist_dir="./storage_vector")
vector_index = load_index_from_storage(
    service_context=service_context, storage_context=storage_context_vector
)

from llama_index.query_engine import KnowledgeGraphQueryEngine

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

nl2kg_query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)

kg_rag_query_engine = kg_index.as_query_engine(
    include_text=False,
    retriever_mode="keyword",
    response_mode="tree_summarize",
)

vector_rag_query_engine = vector_index.as_query_engine()

# graph + vector rag
# import QueryBundle
from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever

from typing import List


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine

# create custom retriever
vector_retriever = VectorIndexRetriever(index=vector_index)
kg_retriever = KGTableRetriever(
    index=kg_index, retriever_mode="keyword", include_text=False
)
custom_retriever = CustomRetriever(vector_retriever, kg_retriever)

# create response synthesizer
response_synthesizer = get_response_synthesizer(
    service_context=service_context,
    response_mode="tree_summarize",
)

graph_vector_rag_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)


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


# write string to file
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


st.title("Graph RAG vs RAG vs NL2Cypher")

(
    tab_code_rag,
    tab_notebook,
    tab_NL2Cypher_vs_GraphRAG,
    tab_Vector_vs_Graph_Vector,
) = st.tabs(
    [
        "Code: Graph RAG",
        "Full Notebook",
        "Demo: NL2Cypher vs Graph RAG",
        "Demo: Vector vs Graph + Vector",
    ]
)


with tab_code_rag:
    st.write(
        "To Create LLM Apps, we could leverage Knowledge Graph in different approaches: **NL2Cypher**, **Graph RAG** and **Graph + Vector RAG**, this Notebook demonstrates the know-how and the comparison between the different approaches."
    )
    st.write(
        "See full notebook for more details and try different approaches online demo on corresponding tabs."
    )
    st.code(body=CODE_KG_RAG, language="python")

with tab_notebook:
    st.write("> Full Notebook")
    st.markdown(
        """

This is the full notebook to demonstrate how to:

- Extract from data sources and build a knowledge graph with LLM and Llama Index, NebulaGraph in 3 lines of code
- QA with NL2Cypher, 3 lines of code
- QA with Graph RAG, 3 lines of code
- QA with Graph + Vector RAG
- Compare the performance of different approaches
        """
    )
    # link to download notebook
    st.markdown(
        """
[Download](https://www.siwei.io/demo-dumps/graph-rag/GraphRAG.ipynb) the notebook.
"""
    )

    components.iframe(
        src="https://www.siwei.io/demo-dumps/graph-rag/GraphRAG.html",
        height=2000,
        width=1000,
        scrolling=True,
    )


with tab_NL2Cypher_vs_GraphRAG:
    st.write("> NL2Cypher vs Graph RAG")

    query_string = st.text_input(
        label="Enter natural language query string", value="Tell me about Peter Quill?"
    )
    col_NL2Cypher, col_GraphRAG = st.columns(2)
    if st.button("Generate Answer with NL2Cypher and Graph RAG"):
        response_NL2Cypher = nl2kg_query_engine.query(query_string)
        response_GraphRAG = kg_rag_query_engine.query(query_string)
        with col_NL2Cypher:
            response = response_NL2Cypher
            graph_query = list(response.metadata.values())[0]["graph_store_query"]
            graph_query = graph_query.replace("WHERE", "\n  WHERE").replace(
                "RETURN", "\nRETURN"
            )
            answer_NL2Cypher = str(response)
            st.markdown(
                f"""
> Query used

```cypher
{graph_query}
```
"""
            )
            st.write("#### Rendered Graph")
            render_query = cypher_to_all_paths(graph_query)
            result = query_nebulagraph(render_query)
            result_df = result_to_df(result)

            # create pyvis graph
            g = create_pyvis_graph(result_df)

            # render with random file name
            graph_html = g.generate_html(f"graph_{random.randint(0, 1000)}.html")

            components.html(graph_html, height=500, scrolling=True)

            st.write(f"*Answer*: {answer_NL2Cypher}")

        with col_GraphRAG:
            response = response_GraphRAG
            answer_GraphRAG = str(response)

            related_entities = list(
                list(response.metadata.values())[0]["kg_rel_map"].keys()
            )
            render_query = f"MATCH p=(n)-[*1..2]-() \n  WHERE id(n) IN {related_entities} \nRETURN p"

            st.markdown(
                f"""
> RAG Subgraph Query(depth=2)

```cypher
{render_query}
```
                """
            )
            st.write("#### Rendered Graph")
            result = query_nebulagraph(render_query)
            result_df = result_to_df(result)

            # create pyvis graph
            g = create_pyvis_graph(result_df)

            # render with random file name
            graph_html = g.generate_html(f"graph_{random.randint(0, 1000)}.html")

            components.html(graph_html, height=500, scrolling=True)

            st.write(f"*Answer*: {answer_GraphRAG}")
        st.write("## Compare the two QA result")
        result = llm.complete(
            f"""
Compare the two QA result on "{query_string}", list the differences between them, to help evalute them. Output in markdown table.

Result from NL2Cypher: {str(response_NL2Cypher)}
---
Result from Graph RAG: {str(response_GraphRAG)}
"""
        )
        st.markdown(result.text)

with tab_Vector_vs_Graph_Vector:
    st.write("> Vector RAG vs Graph + Vector RAG")
    query_string = st.text_input(
        label="Type the question to answer", value="Tell me about Peter Quill?"
    )
    col_VectorRAG, col_GraphVectorRAG = st.columns(2)
    if st.button("Generate Answer with Vector and Graph + Vector"):
        response_VectorRAG = vector_rag_query_engine.query(query_string)
        response_GraphVectorRAG = graph_vector_rag_query_engine.query(query_string)
        with col_VectorRAG:
            response = response_VectorRAG
            answer_VectorRAG = str(response)
            st.write(f"*Answer*: {answer_VectorRAG}")

        with col_GraphVectorRAG:
            response = response_GraphVectorRAG
            answer_GraphVectorRAG = str(response)
            st.write(f"*Answer*: {answer_GraphVectorRAG}")

        st.write("## Compare the two QA result")
        st.markdown(
            llm.complete(
                f"""
Compare the two QA result on "{query_string}", list the differences between them, to help evalute them. Output in markdown table.

Result from Vector RAG: {str(response_VectorRAG)}
---
Result from Graph+Vector RAG: {str(response_GraphVectorRAG)}
"""
            ).text
        )
