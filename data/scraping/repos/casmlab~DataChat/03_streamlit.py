#!/usr/bin/env python
# coding: utf-8

# Reference: https://github.com/tomasonjo/NeoGPT-Explorer/tree/main/streamlit/src
# Reference: https://github.com/ChrisDelClea/streamlit-agraph

### driver
from neo4j import GraphDatabase

host = 'bolt://localhost:7687'
user = 'neo4j'
password = "XXXX"
driver = GraphDatabase.driver(host, auth=(user, password))


def read_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        response = [r.values()[0] for r in result]
        return response

def read_graph(messages):
    # create
    nodes = []
    edges = []
    for name in messages:
        name = name.split("LINK:")[0].strip()
        nodes.append({"id":name, "label":"Dataset"})
        new_query = """
        MATCH (a:Dataset)-[r]-(connectedNodes)
        WHERE a.name = $datasetName
        RETURN labels(connectedNodes) AS NodeType, connectedNodes.name AS ConnectedNode, type(r) AS RelationType  
        """
        with driver.session() as session:
            result = session.run(new_query, {"datasetName":name})
            for rel in result:
                this_rel = {"source":name, "label":rel["RelationType"], "target":rel["ConnectedNode"]}
                edges.append(this_rel)
                nodes.append({"id":rel["ConnectedNode"], "label":rel["NodeType"][0]})
    # format (also remove duplicate)
    formatted_nodes, formatted_edges = [], []
    nodes_seen = set()
    nodes_color = {"Dataset":"#4e79a7",
                    "Term":"#bab0ac",
                    "Location":"#edc948",
                    "Series":"#b07aa1",
                    "Funder":"#59a14f",
                    "Owner":"#e15759",
                    "Publication":"#f28e2b"}
    for node in nodes:
        if node["id"] not in nodes_seen:
            nodes_seen.add(node["id"])
            formatted_nodes.append(Node(id=node["id"], label=node["label"], size=25, shape="dot", color=nodes_color[node["label"]])) 
    for edge in edges:
        formatted_edges.append(Edge(source=edge["source"], label=["label"], target=edge["target"])) 

    return formatted_nodes, formatted_edges


### example cypher
examples = """
# What are the latest datasets?
MATCH (a:Dataset) RETURN a.name + " LINK: " + a.url AS response ORDER BY a.date DESC LIMIT 3
# What are the most cited datasets?
MATCH (a:Dataset) RETURN a.name + " LINK: " + a.url AS response ORDER BY a.dataRefCount DESC LIMIT 3
# What are the most used datasets?
MATCH (a:Dataset) RETURN a.name + " LINK: " +  a.url AS response ORDER BY a.dataUserCount DESC LIMIT 3
# What are the latest datasets not owned by ICPSR?
MATCH (a:Dataset) WHERE a.owner <> 'ICPSR' RETURN a.name + " LINK: " +  a.url AS response ORDER BY a.date DESC LIMIT 3
# What are the latest datasets about alcohol?
MATCH (a:Dataset) WHERE a.name CONTAINS 'alcohol' RETURN a.name + " LINK: " +  a.url AS response ORDER BY a.date DESC LIMIT 3
# What are the latest datasets that mention alcohol?
MATCH (a:Dataset)-[:HAS_TERM]->(t:Term) WHERE t.name CONTAINS 'alcohol' RETURN a.name + " LINK: " +  a.url AS response ORDER BY a.date DESC LIMIT 3
# What are the latest datasets in the United States Historical Election Returns Series?
MATCH (a:Dataset)-[:HAS_SERIES]->(s:Series) WHERE s.name = 'United States Historical Election Returns Series' RETURN a.name + " LINK: " +  a.url AS response ORDER BY a.date DESC LIMIT 3
# What are the latest datasets onwed by ICPSR?
MATCH (a:Dataset)-[:HAS_OWNER]->(o:Owner) WHERE o.name = 'ICPSR' RETURN a.name + " LINK: " +  a.url AS response ORDER BY a.date DESC LIMIT 3
# What are the latest datasets funder by National Science Foundation?
MATCH (a:Dataset)-[:HAS_FUNDER]->(f:Funder) WHERE f.name = "National Science Foundation" RETURN a.name + " LINK: " +  a.url AS response ORDER BY a.date DESC LIMIT 3
# What are the latest datasets funder by government?
MATCH (a:Dataset)-[:HAS_FUNDER]->(f:Funder) WHERE f.type = "government" RETURN a.name + " LINK: " +  a.url AS response ORDER BY a.date DESC LIMIT 3
# What are the latest datasets that include data from United States?
MATCH (a:Dataset)-[:HAS_LOCATION]->(l:Location) WHERE (l.name = "United States" OR l.name = "U.S.") RETURN a.name + " LINK: " +  a.url AS response ORDER BY a.date DESC LIMIT 3
# What are the latest datasets that include country level data?
MATCH (a:Dataset)-[:HAS_LOCATION]->(l:Location) WHERE l.type = "country" RETURN a.name + " LINK: " +  a.url AS response ORDER BY a.date DESC LIMIT 3
"""

import os
import openai
import streamlit as st
from streamlit_chat import message
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_agraph.config import Config, ConfigBuilder

openai.api_key = 'XXXX'

### eg
# prompt_eg = "What are the earlest datasets?"
# prompt_input=examples_eg + "\n#" + prompt_eg
# completions = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     max_tokens=1000,
#     n=1,
#     stop=None,
#     temperature=0.5,
#     messages=[{"role": "user", "content": prompt_input}]
# )
# cypher_query = completions['choices'][0]['message']['content']
# message = read_query(cypher_query)
# print(message)
# print(cypher_query)


st.title("DataChat: Chat with ICPSR Datasets")

def generate_response(prompt):
    #prompt_eg = "What are the earlest datasets?"
    prompt_input=examples + "\n#" + prompt
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt_input}]
    )
    cypher_query = completions['choices'][0]['message']['content']
    message = read_query(cypher_query) # list of string
    nodes, edges = read_graph(message)
    return message, nodes, edges, cypher_query

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'graph' not in st.session_state:
    st.session_state['graph'] = []

def get_text():
    input_text = st.text_input(
        "Feel free to ask a question about ICPSR datasets", "", key="input")
    return input_text


tab1, tab2 = st.tabs(["🤖 DataChatBot", "🌐 DataChatViz"])

tab1.subheader("Datasets with links")
tab2.subheader("Interactive graphs")

user_input = get_text()
if user_input:
    output, nodes, edges, cypher_query = generate_response(user_input)

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col2:
            another_placeholder = st.empty()
        with col1:
            placeholder = st.empty()

        # store the output
        st.session_state.past.append(user_input)
        st.session_state.generated.append((output, cypher_query))

        # Message placeholder
        with placeholder.container():
            if st.session_state['generated']:
                message(st.session_state['past'][-1],
                        is_user=True, key=str(-1) + '_user')
                for j, text in enumerate(st.session_state['generated'][-1][0]):
                    message(text, key=str(-1) + str(j))

        # Generated Cypher statements
        with another_placeholder.container():
            if st.session_state['generated']:
                st.text_area("Generated Cypher statement",
                             st.session_state['generated'][-1][1], height=240)

    with tab2:
        config = Config(width=750,
                        height=950,
                        directed=True, 
                        physics=True, 
                        hierarchical=False,
                        # **kwargs
                        )
        st.session_state.graph.append(agraph(nodes=nodes, edges=edges, config=config))

###### Saved earlier version

# tab1, tab2 = st.tabs(["🤖 DataChatBot", "🌐 DataChatViz"])

# tab1.subheader("DataChatBot")
# tab2.subheader("DataChatViz")

# col1, col2 = st.columns([2, 1])

# with col2:
#     another_placeholder = st.empty()
# with col1:
#     placeholder = st.empty()
# user_input = get_text()

# config = Config(width=750,
#                 height=950,
#                 directed=True, 
#                 physics=True, 
#                 hierarchical=False,
#                 # **kwargs
#                 )

# if user_input:
#     output, nodes, edges, cypher_query = generate_response(user_input)
#     # store the output
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append((output, cypher_query))
#     st.session_state.graph.append(agraph(nodes=nodes, edges=edges, config=config))

# # Message placeholder
# with placeholder.container():
#     if st.session_state['generated']:
#         message(st.session_state['past'][-1],
#                 is_user=True, key=str(-1) + '_user')
#         for j, text in enumerate(st.session_state['generated'][-1][0]):
#             message(text, key=str(-1) + str(j))

# # Generated Cypher statements
# with another_placeholder.container():
#     if st.session_state['generated']:
#         st.text_area("Generated Cypher statement",
#                      st.session_state['generated'][-1][1], height=240)



