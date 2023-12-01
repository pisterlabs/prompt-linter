import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import openai
import plotly.graph_objects as go
import json
import networkx as nx
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pyvis.network import Network
import random
import datetime
import re

# OpenAI API key
openai.api_key = "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"


def plot_pyvis_network_graph(graph_data, save_name="example_kg"):
    # Create a NetworkX graph
    G = nx.Graph()

    for node in graph_data["nodes"]:
        G.add_node(
            node["id"], label=node["label"], title=node["tooltip"]
        )  # Using "title" for tooltips

    for edge in graph_data["edges"]:
        G.add_edge(
            edge["source"], edge["target"], label=edge["label"]
        )  # Removing tooltips from edges

    # Create a Pyvis network
    net = Network(notebook=True, height="600px", width="60%")
    # Define the options as a Python dictionary
    options = {
        "nodes": {"font": {"size": 10}},
        "edges": {"font": {"size": 7}, "arrows": {"to": {"enabled": True}}},
        "physics": {
            "enabled": True,
            "repulsion": {
                "centralGravity": 0.0,  # Set to 0 to make nodes repel from each other
                "springLength": 100,
                "springConstant": 0.05,
                "nodeDistance": 100,
            },
        },
        "interaction": {"hover": True},
    }

    # Convert the options dictionary to a JSON string
    options_str = json.dumps(options)

    # Set the options for improved visualization
    net.set_options(options_str)

    # Load the NetworkX graph into the Pyvis network
    net.from_nx(G)

    # Show the interactive graph and save it to an HTML file
    net.show(f"{save_name}.html")


def plot_springy_network_graph(graph_data):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges from the provided graph_data
    for node in graph_data['nodes']:
        G.add_node(node['id'], label=node['label'], tooltip=node['tooltip'])

    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'], label=edge['label'], tooltip=edge['tooltip'])

    # Generate a layout using a spring layout algorithm
    pos = nx.spring_layout(G, seed=42)

    # Create a Plotly subplot
    fig = make_subplots(rows=1, cols=1)

    # Extract node positions for Plotly
    node_x = []
    node_y = []
    for node_id, coords in pos.items():
        node_x.append(coords[0])
        node_y.append(coords[1])

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        source, target, _ = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend((x0, x1, None))
        edge_y.extend((y0, y1, None))
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(thickness=15, title='Node Connections'),
            line_width=2
        )
    )

    node_texts = []
    for node_id in G.nodes:
        node = G.nodes[node_id]
        label = node['label']
        tooltip = node['tooltip']
        node_texts.append(f'Label: {label}<br>Tooltip: {tooltip}')

    node_trace.text = node_texts

    # Add node and edge traces to the figure
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    # Customize the layout of the plot
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        title='Movable Springy Network Graph',
        title_x=0.5
    )

    # Show the plot
    fig.show()


def generate_knowledge_graph(text):
    text = re.sub(r'\n+', r'\n', text)

    prompt = """Act As an API endpoint, your task is to process the given text and generate a Python dictionary representing a knowledge graph based on the content. Avoid starting your response with phrases like "Here's a Python dictionary..." or ending with explanatory text. Your output should be a Python dictionary.
Text: {text_query}
> Response Structure example:
{{
    'nodes': [
        {{'id': '1',
            'label': <some short text name for the node>,
            'tooltip': <some longer text description for the node>,}},
        ...
    ],
    'edges': [
        {{'id': '1',
            'label': <some short text name for the relation between the nodes e.g "is a" or "has" or any other complex relations too>,
            'tooltip': <some longer text description for the relation between the nodes>,
            'source': '1',
            'target': '2',}},
        ...
    ]
}}
""".format(text_query=text)
        
    # check if cache folder exists
    if not os.path.exists("cache"):
        os.makedirs("cache")

    # save prompt to file
    tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    with open(f"cache/prompt_{tag}.txt", "w") as f:
        f.write(prompt)

    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
    
    # save response to json file
    with open(f"cache/response_{tag}.json", "w") as f:
        json.dump(completion, f)
    
    # parse response
    kg = eval(completion.choices[0]['message']['content'])

    # check if kg folder exists
    if not os.path.exists("kg"):
        os.makedirs("kg")

    # save kg to file
    with open(f"kg/kg_{tag}.json", "w") as f:
        json.dump(kg, f)


    # check plots folder exists
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plot_pyvis_network_graph(kg, save_name=f"plots/pyvis_temp")

    return kg

if __name__ == "__main__":
    text_query = """LlamaIndex provides the following tools:

Data connectors ingest your existing data from their native source and format. These could be APIs, PDFs, SQL, and (much) more.

Data indexes structure your data in intermediate representations that are easy and performant for LLMs to consume.

Engines provide natural language access to your data. For example: - Query engines are powerful retrieval interfaces for knowledge-augmented output. - Chat engines are conversational interfaces for multi-message, “back and forth” interactions with your data.

Data agents are LLM-powered knowledge workers augmented by tools, from simple helper functions to API integrations and more.

Application integrations tie LlamaIndex back into the rest of your ecosystem. This could be LangChain, Flask, Docker, ChatGPT, or… anything else!"""

    
    kg = generate_knowledge_graph(text_query)
    print(kg)