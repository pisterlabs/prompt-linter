import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

import instructor
import openai
import streamlit as st
from graphviz import Digraph
from pydantic import ValidationError
from pyvis.network import Network

from models import KnowledgeGraph

logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if OPENAI_API_KEY is None:
    st.error("OPENAI API key Missing")
    st.stop()

openai.api_key = OPENAI_API_KEY


# Adds response_model to ChatCompletion
# Allows the return of Pydantic model rather than raw JSON
instructor.patch()


def loading_animation(stop_event):
    chars = "|/-\\"
    prefix = "Generating graph... "
    sys.stdout.write(prefix)
    sys.stdout.flush()
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write("\r" + prefix + chars[idx % len(chars)])
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write(
        "\r" + prefix + "Done!    "
    )  # extra spaces to ensure clearing any leftover chars
    sys.stdout.flush()


def generate_file_path(topic, directory, extension):
    filepath = directory / f"{topic.replace(' ', '_').lower()}.{extension}"
    return filepath


def create_prompt(input: str, depth: str) -> str:
    depth_description = {
        "overview": "Provide a high-level overview focusing on the core concepts and their direct relationships.",
        "deep": """
        Delve into a multi-layered detailed exploration of the topic. For each main node or concept:
            - Branch out to its primary sub-concepts or details (1st layer).
            - For each of these primary sub-concepts, further expand to their sub-details or related concepts (2nd layer).
        Ensure at least three layers of depth in the graph.
        """,
    }
    return f"""
    Construct a knowledge graph for the topic '{input}'. {depth_description.get(depth, '')}

    Key guidelines:
    - Avoid generic placeholders like "primary node" or "secondary node". Every node should represent a concrete concept or detail.
    - Describe the significance and relationship of each node to the overarching topic.
    - Define relationships with precision. For instance, specify if a node "is a type of", "results in", "is used for", "is an example of", etc.
    - Assign each node an 'id', 'label', and 'color' based on its importance or depth in the topic (e.g., fundamental concepts might be red, primary sub-concepts orange, secondary sub-concepts blue, and so on).
    - If there's ambiguity in any node's placement or relevance, provide a brief reasoning or source for clarity.

    Aim for a {depth}-level understanding, ensuring that the essence of the topic is captured from all angles and depths.
    """


def generate_graph(input, depth, max_retries=3) -> KnowledgeGraph:
    stop_event = threading.Event()
    t = threading.Thread(target=loading_animation, args=(stop_event,))
    t.start()

    for _ in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": create_prompt(input, depth)}],
                response_model=KnowledgeGraph,
            )
            stop_event.set()
            return response
        except ValidationError:
            logging.warning(f"\nRetry attempt {_ + 1} failed. Trying again...")
            continue
    stop_event.set()


def visualize_knowledge_graph_interactive(kg, name, directory):
    filepath = generate_file_path(name, directory, "html")
    nt = Network(notebook=True, width="100%", height="800px", cdn_resources="in_line")

    for node in kg.nodes:
        nt.add_node(node.id, node.label, title=node.label, color=node.color)
    for edge in kg.edges:
        nt.add_edge(edge.source, edge.target, label=edge.label, color=edge.color)

    nt.set_options(
        """
    var options = {
    "nodes": {
        "borderWidth": 2,
        "size": 30,
        "color": {
        "border": "rgba(0,217,255,1)",
        "background": "rgba(0,217,255,1)"
        },
        "font": {
        "color": "black"
        }
    },
    "edges": {
        "color": {
        "color": "rgba(128,128,128,1)",
        "inherit": false
        },
        "smooth": false
    },
    "physics": {
        "enabled": true,
        "barnesHut": {
        "gravitationalConstant": -4000,
        "centralGravity": 0.1,
        "springLength": 150,
        "springConstant": 0.02,
        "damping": 0.09,
        "avoidOverlap": 0.5
        },
        "maxVelocity": 146.5,
        "minVelocity": 0.1,
        "solver": "barnesHut",
        "timestep": 0.35,
        "adaptiveTimestep": true
    }
    }
    """
    )

    nt.save_graph(str(filepath))
    logging.info(f"Saved interactive graph to: {filepath}")
    return filepath


def visualize_knowledge_graph(kg, name, directory):
    filepath = generate_file_path(name, directory, "svg")
    dot = Digraph(comment=name, format="svg")

    # Graph aesthetics
    dot.attr(
        bgcolor="#F5F5F5",
        rankdir="TB",
        nodesep="0.5",
        ranksep="1",
        overlap="false",
        outputorder="edgesfirst",
        pad="0.5",
    )

    # Node aesthetics
    for node in kg.nodes:
        dot.node(
            str(node.id),
            node.label,
            shape="ellipse",
            fontsize="12",
            color=node.color,
            gradientangle="90",
        )

    # Edge aesthetics
    for edge in kg.edges:
        dot.edge(
            str(edge.source),
            str(edge.target),
            label=edge.label,
            color="#A9A9A9",
            arrowsize="0.5",
            penwidth="1.5",
        )

    dot.render(filename=str(filepath.stem), directory=str(filepath.parent))
    logging.info(f"Saved graph to: {filepath}")
    return filepath


def list_edge_nodes(kg: KnowledgeGraph):
    nodes_with_edges = {edge.source for edge in kg.edges}

    # Filter out nodes that don't have outgoing edges
    return [node for node in kg.nodes if node.id not in nodes_with_edges]


def get_depth_from_user():
    depth = (
        input("Enter the depth of the graph you want to generate (overview/deep): ")
        .strip()
        .lower()
    )
    if depth not in ["overview", "deep"]:
        print("Invalid depth. Please enter either 'overview' or 'deep'.")
        return get_depth_from_user()
    return depth


def get_topic_from_user():
    topic = input("Enter the topic you want to learn about: ")
    if not topic:
        print("Invalid topic. Please enter a valid topic.")
        return get_topic_from_user()
    return topic


def show_graph(topic, depth, directory_path):
    sanitized_topic = topic.replace(" ", "_").lower()

    svg_path = directory_path / f"{sanitized_topic}.svg"
    html_path = directory_path / f"{sanitized_topic}.html"

    if not svg_path.exists() and not html_path.exists():
        graph: KnowledgeGraph = generate_graph(topic, depth)

        save_edge_nodes(graph, directory=directory_path)
        visualize_knowledge_graph(graph, name=topic, directory=directory_path)
        visualize_knowledge_graph_interactive(
            graph, name=topic, directory=directory_path
        )

    choice = input("\nDo you want an interactive graph? (yes/no): ").strip().lower()
    if choice == "yes":
        webbrowser.open("file://" + str(html_path.absolute()))
    else:
        webbrowser.open("file://" + str(svg_path.absolute()))

    return graph


def save_edge_nodes(graph, directory):
    edge_nodes = list_edge_nodes(graph)
    filepath = generate_file_path("edges", directory, "txt")
    with open(filepath, "w") as f:
        for node in edge_nodes:
            f.write(node.label + "\n")


def user_interaction():
    topic = get_topic_from_user()
    depth = get_depth_from_user()
    directory_path = Path("graphs") / topic.replace(" ", "_").lower().lower()

    graph = show_graph(topic, depth, directory_path)
    edge_nodes = list_edge_nodes(graph)

    while edge_nodes:
        print("\nMain concepts from the generated graph:")
        for idx, node in enumerate(edge_nodes, 1):
            print(f"{idx}. {node.label}")

        try:
            choice = int(
                input(
                    "Which concept would you like to dive deeper into? (Enter the number or 0 to skip): "
                )
            )
            if choice < 0 or choice > len(edge_nodes):
                raise ValueError
        except ValueError:
            return print(
                "Invalid choice. Please select a valid number from the list or 0 to skip."
            )

        if choice != 0:
            deeper_topic = edge_nodes[choice - 1].label
            deeper_depth = "deep"
            graph = show_graph(deeper_topic, deeper_depth, directory_path)
            edge_nodes = list_edge_nodes(graph)
        else:
            return print(
                "Thank you for using the knowledge graph generator. Have a great day!"
            )


if __name__ == "__main__":
    user_interaction()
