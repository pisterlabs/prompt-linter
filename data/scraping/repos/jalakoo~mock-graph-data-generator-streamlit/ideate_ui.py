# Original ChatGPT ideation tab - needs updating to work

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import logging
import openai
import json
import random
import base64
import ast

# Yes yes - move all the non-ui stuff into a controller or something already

CHATGPT_KEY = "chatgpt"

def arrows_uri(input: str | dict) -> str:
    """
    Generates a URI for an arrows app visualization from a json object. WARNING! May overwrite existing arrows drawing.

    Args:
        input: A dictionary or string representing an arrows compatible .json configuration

    Returns:
        A string URI for an arrows app visualization
    """

    # Convert dict to string if needed
    if isinstance(input, dict):
        input = json.dumps(input)

    # Convert the diction object into a base 64 json string
    b = input.encode('utf-8')
    base64_str = base64.b64encode(b).decode('utf-8')

    result = f"https://arrows.app/#/import/json={base64_str}"

    logging.debug(f'\n\nOutput arrows uri from {input} with base64 JSON: \n{result}')

    return result


def arrows_dictionary(nodes: list[Node], edges: list[Edge], name: str = "GraphGPT Generated Model") -> dict:
    """
    Generates an arrows.app compatible .json file from agraph nodes and edges

    Args:
        nodes: List of agraph Nodes
        edges: List of agraph Edges

    Returns:
        A dictionary matching arrows .json schema
    """
    result_nodes = []
    result_relationships = []
    for n in nodes:
        random_x = round(random.uniform(-600, 600), 14)
        random_y = round(random.uniform(-600, 600), 14)
        result_nodes.append({
            "id": n.id,
            "position":{
                "x" : random_x,
                "y": random_y
            },
            "caption": n.label,
            "style":{},
            "labels":[],
            "properties":{}
        })
    for idx, e in enumerate(edges):
        logging.debug(f'Processing edge to relationships: {e.__dict__}')
        ns = e.source
        nt = e.to
        type = e.label.replace(" ", "_")
        result_relationships.append(
            {
                "id":f"n{idx}",
                "type":type,
                "fromId":ns,
                "toId":nt,
                "style":{},
                "properties":{}
            }
        )
    result = {
        "graph": {
            "style": {
            "font-family": "sans-serif",
            "background-color": "#ffffff",
            "background-image": "",
            "background-size": "100%",
            "node-color": "#ffffff",
            "border-width": 4,
            "border-color": "#000000",
            "radius": 50,
            "node-padding": 5,
            "node-margin": 2,
            "outside-position": "auto",
            "node-icon-image": "",
            "node-background-image": "",
            "icon-position": "inside",
            "icon-size": 64,
            "caption-position": "inside",
            "caption-max-width": 200,
            "caption-color": "#000000",
            "caption-font-size": 50,
            "caption-font-weight": "normal",
            "label-position": "inside",
            "label-display": "pill",
            "label-color": "#000000",
            "label-background-color": "#ffffff",
            "label-border-color": "#000000",
            "label-border-width": 4,
            "label-font-size": 40,
            "label-padding": 5,
            "label-margin": 4,
            "directionality": "directed",
            "detail-position": "inline",
            "detail-orientation": "parallel",
            "arrow-width": 5,
            "arrow-color": "#000000",
            "margin-start": 5,
            "margin-end": 5,
            "margin-peer": 20,
            "attachment-start": "normal",
            "attachment-end": "normal",
            "relationship-icon-image": "",
            "type-color": "#000000",
            "type-background-color": "#ffffff",
            "type-border-color": "#000000",
            "type-border-width": 0,
            "type-font-size": 16,
            "type-padding": 5,
            "property-position": "outside",
            "property-alignment": "colon",
            "property-color": "#000000",
            "property-font-size": 16,
            "property-font-weight": "normal"
            },
            "nodes":result_nodes,
            "relationships":result_relationships,
            "diagramName": name
        }
    }

    logging.debug(f'\n\nProcessed incoming nodes: {nodes}, edges: {edges} to:\n {result}')
    
    return result


def triples_prompt(prompt: str)-> str:
    # Full prompt string to query openai with and finasse expected response
    full_prompt = f"""
    Given a prompt, extrapolate as many relationships as possible from it and provide a list of updates.

    If an update is a relationship, provide [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.

    Each relationship must have 3 items in the list.
    Limit the number of relationships to 12.
    Return only the data, do not explain.
    Only return a list of 3 item lists.

    For example, the prompt: `Alice is Bob's roommate` should return [["Alice", "roommate", "Bob"]]

    prompt: {prompt}
    """
    return full_prompt


def agraph_nodes_edges(response: str | list) -> tuple[list[Node], list[Edge]]:
    """
    Converts an openai response into agraph nodes and relationships

    Args:
        response: String or list in the format of [["node_id_string", "edge_id_string", "another_node_id_string"],...]
    
    Returns:
        A tuple of agraph nodes in a list and agraph edges in a list

    Raises:
        ...
    """
    logging.debug(f'agraph_nodes_edges() response recieved: {response}')
    # Response will be a list of 3 item tuples

    # Convert to list of lists - if needed
    if isinstance(response, str):
        try:
            answers = json.loads(response)
        except:
            logging.debug(f'Unable to parse string to list with json.loads. Attempting ast...')
            answers = ast.literal_eval(response)

    elif isinstance(response, list):
        answers = response
    else:
        raise ValueError(f'Response is not a string or list. Response: {response}')

    logging.debug(f'JSON parsed: {answers}')
    nodes = set()
    result_edges = []
    for item in answers:
        # Each should be a tuple of 3 items, node-edge-node
        n1 = item[0]
        r = item[1]
        n2 = item[2]

        # Standardize casing
        r = r.upper()
        n1 = n1.title()
        n2 = n2.title()

        nodes.add(n1)
        nodes.add(n2) 

        edge = Edge(source=n1, target=n2, label=r)
        result_edges.append(edge)

    result_nodes = []
    for node_label in list(nodes):
        node = Node(id=node_label, label=node_label)
        result_nodes.append(node)

    logging.debug(f'Nodes returning: {result_nodes}')
    logging.debug(f'Edges returning: {result_edges}')

    return result_nodes, result_edges

@st.cache_data
def generate_openai_response(prompt)-> str:
    # TODO: Make this configurable
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    # TODO: Validate reponse
    content = response.choices[0].message.content
    logging.debug(f'OpenAI Response: {response}, type: {type(content)}')
    return content
    
def agraph_from_sample(prompt: str):
    # TODO: Pull from a file of samples
    openai_response = '[["Sharks", "eat", "big fish"], ["Big fish", "eat", "small fish"], ["Small fish", "eat", "bugs"]]'
    nodes, edges = agraph_nodes_edges(openai_response)
    config = Config(height=400, width=1000, directed=True)

    if nodes is not None:
        agraph(nodes=nodes, 
            edges=edges, 
            config=config) 

def ideate_ui():

    st.markdown("Use a variation of Varun Shenoy's original [GraphGPT](https://graphgpt.vercel.app) to convert a natural language description into a graph data model")

    # OPENAI TEXTFIELD
    new_open_ai_key = st.text_input(f'OpenAI KEY', type="password", value=st.session_state["OPENAI_API_KEY"])

    # Set openAI key
    openai.api_key = new_open_ai_key

    # Display prompt for user input
    sample_prompt = "Sharks eat big fish. Big fish eat small fish. Small fish eat bugs."
    if st.button('Load Sample', key="graphgpt_sample"):
        st.session_state["SAMPLE_PROMPT"] = sample_prompt

    prompt = st.text_area("Prompt", value=st.session_state["SAMPLE_PROMPT"])
    if prompt is None or prompt == "":
        return

    if prompt == sample_prompt:
        # Load vetted response to save on hitting openai for the same thing
        response = [["Sharks", "eat", "big fish"], ["Big fish", "eat", "small fish"], ["Small fish", "eat", "bugs"]]
    else: 
        # Send completion request to openAI
        full_prompt = triples_prompt(prompt)
        response = generate_openai_response(full_prompt) 

    # Convert response to agraph nodes and edges
    nodes, edges = agraph_nodes_edges(response)

    # Configure and display agraph
    config = Config(width=1000, height=400, directed=True)
    if nodes is None:
        return
    
    # Display data
    st.write('Graph Viewer')
    agraph(nodes=nodes,  
        edges=edges, 
        config=config)
    
    # For displaying JSON schema. This can be quite long though
    # st.write('JSON Representation')
    # arrows_str = json.dumps(arrows_dict, indent=4)
    # st.code(arrows_str)

    # Prep arrows compatible dictioary for button options
    arrows_dict = arrows_dictionary(nodes, edges)

    b1, b2, b3 = st.columns([1,1,3])
    with b1:
        if st.button("Edit in Arrows"):
            # Prep arrows compatible json
            uri = arrows_uri(arrows_dict)
            st.session_state["ARROWS_URI"] = uri
            st.warning("Close and reopen 'Arrows Data Modeler' to refresh")
    with b2:
        if st.button("Push to Generator"):
            st.session_state["ARROWS_DICT"] = arrows_dict

# def agraph_sample():
#     # Agraph
#     nodes = []
#     edges = []
#     nodes.append( Node(id="Spiderman", 
#                     label="Peter Parker", 
#                     size=25, 
#                     shape="circularImage",
#                     image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_spiderman.png") 
#                 ) # includes **kwargs
#     nodes.append( Node(id="Captain_Marvel", 
#                     size=25,
#                     shape="circularImage",
#                     image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_captainmarvel.png") 
#                 )
#     edges.append( Edge(source="Captain_Marvel", 
#                     label="friend_of", 
#                     target="Spiderman", 
#                     # **kwargs
#                     ) 
#                 ) 

#     config = Config(width=750,
#                     height=950,
#                     directed=True, 
#                     physics=True, 
#                     hierarchical=False,
#                     # **kwargs
#                     )

#     agraph(nodes=nodes, 
#             edges=edges, 
#             config=config)