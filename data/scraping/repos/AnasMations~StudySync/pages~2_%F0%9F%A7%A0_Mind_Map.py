#================================================================ Importing ===================================================================
from __future__ import annotations
import re
from typing import Optional, Tuple, List, Literal
import streamlit as st
import os
import openai
from dataclasses import dataclass, asdict
from textwrap import dedent
from streamlit_agraph import agraph, Node, Edge, Config
from dotenv import load_dotenv, find_dotenv

#======================================================= basic config =======================================================================
# set title of page (will be seen in tab) and the width
st.set_page_config(page_title="AI Mindmap", page_icon="https://github.com/AnasMations/StudySync/blob/36ae1cad78544b8a07239d1cefa141a59f6305c8/img/icon.png?raw=true", layout="wide")

COLOR = "#6dbbbd"
FOCUS_COLOR = "#b9359a"
EDGE_COLOR = "#b9359a"

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
col1, col2= st.columns([1, 2])

#=================================================== basic functions of gpt ========================================================================
@dataclass
class Message:
    """A class that represents a message in a ChatGPT conversation.
    """
    content: str
    role: Literal["user", "system", "assistant"]

    # is a built-in method for dataclasses
    # called after the __init__ method
    def __post_init__(self):
        self.content = dedent(self.content).strip()

START_CONVERSATION = [
    Message("""
        You are a useful mind map/undirected graph-generating AI that can generate mind maps
        based on any input or instructions.
    """, role="system"),
    Message("""
        You have the ability to perform the following actions given a request
        to construct or modify a mind map/graph:

        1. add(node1, node2) - add an edge between node1 and node2
        2. delete(node1, node2) - delete the edge between node1 and node2
        3. delete(node1) - deletes every edge connected to node1

        Note that the graph is undirected and thus the order of the nodes does not matter
        and duplicates will be ignored. Another important note: the graph should be sparse,
        with many nodes and few edges from each node. Too many edges will make it difficult 
        to understand and hard to read. The answer should only include the actions to perform, 
        nothing else. If the instructions are vague or even if only a single word is provided, 
        still generate a graph of multiple nodes and edges that that could makes sense in the 
        situation. Remember to think step by step and debate pros and cons before settling on 
        an answer to accomplish the request as well as possible.

        Here is my first request: Add a mind map about machine learning.
    """, role="user"),
    Message("""
        add("Machine learning","AI")
        add("Machine learning", "Reinforcement learning")
        add("Machine learning", "Supervised learning")
        add("Machine learning", "Unsupervised learning")
        add("Supervised learning", "Regression")
        add("Supervised learning", "Classification")
        add("Unsupervised learning", "Clustering")
        add("Unsupervised learning", "Anomaly Detection")
        add("Unsupervised learning", "Dimensionality Reduction")
        add("Unsupervised learning", "Association Rule Learning")
        add("Clustering", "K-means")
        add("Classification", "Logistic Regression")
        add("Reinforcement learning", "Proximal Policy Optimization")
        add("Reinforcement learning", "Q-learning")
    """, role="assistant"),
    Message("""
        Remove the parts about reinforcement learning and K-means.
    """, role="user"),
    Message("""
        delete("Reinforcement learning")
        delete("Clustering", "K-means")
    """, role="assistant")
]

def ask_chatgpt(conversation: List[Message]) -> Tuple[str, List[Message]]:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # asdict comes from `from dataclasses import asdict`
        messages=[asdict(c) for c in conversation]
    )
    # turn into a Message object
    msg = Message(**response["choices"][0]["message"])
    # return the text output and the new conversation
    return msg.content, conversation + [msg]
#========================================================= Mindmap ==============================================================================
class MindMap:
    """A class that represents a mind map as a graph.
    """
    
    def __init__(self, edges: Optional[List[Tuple[str, str]]]=None, nodes: Optional[List[str]]=None) -> None:
        self.edges = [] if edges is None else edges
        self.nodes = [] if nodes is None else nodes
        self.save()

    @classmethod
    def load(cls) -> MindMap:
        """Load mindmap from session state if it exists
        
        Returns: Mindmap
        """
        if "mindmap" in st.session_state:
            return st.session_state["mindmap"]
        return cls()

    def save(self) -> None:
        # save to session state
        st.session_state["mindmap"] = self

    def is_empty(self) -> bool:
        return len(self.edges) == 0
    
    def ask_for_initial_graph(self, query: str) -> None:
        """Ask GPT-3 to construct a graph from scrach.

        Args:
            query (str): The query to ask GPT-3 about.

        Returns:
            str: The output from GPT-3.
        """

        conversation = START_CONVERSATION + [
            Message(f"""
                Great, now ignore all previous nodes and restart from scratch. I now want you do the following:    

                {query}
            """, role="user")
        ]

        output, self.conversation = ask_chatgpt(conversation)
        # replace=True to restart
        self.parse_and_include_edges(output, replace=True)

    def ask_for_extended_graph(self, selected_node: Optional[str]=None, text: Optional[str]=None) -> None:
        """Cached helper function to ask GPT-3 to extend the graph.

        Args:
            query (str): query to ask GPT-3 about
            edges_as_text (str): edges formatted as text

        Returns:
            str: GPT-3 output
        """

        # do nothing
        if (selected_node is None and text is None):
            return

        # change description depending on if a node
        # was selected or a text description was given
        #
        # note that the conversation is copied (shallowly) instead
        # of modified in place. The reason for this is that if
        # the chatgpt call fails self.conversation will not
        # be updated
        if selected_node is not None:
            # prepend a description that this node
            # should be extended
            conversation = self.conversation + [
                Message(f"""
                    add new edges to new nodes, starting from the node "{selected_node}"
                """, role="user")
            ]
            st.session_state.last_expanded = selected_node
        else:
            # just provide the description
            conversation = self.conversation + [Message(text, role="user")]

        # now self.conversation is updated
        output, self.conversation = ask_chatgpt(conversation)
        self.parse_and_include_edges(output, replace=False)

    def parse_and_include_edges(self, output: str, replace: bool=True) -> None:
        """Parse output from LLM (GPT-3) and include the edges in the graph.

        Args:
            output (str): output from LLM (GPT-3) to be parsed
            replace (bool, optional): if True, replace all edges with the new ones, 
                otherwise add to existing edges. Defaults to True.
        """

        # Regex patterns
        pattern1 = r'(add|delete)\("([^()"]+)",\s*"([^()"]+)"\)'
        pattern2 = r'(delete)\("([^()"]+)"\)'

        # Find all matches in the text
        matches = re.findall(pattern1, output) + re.findall(pattern2, output)

        new_edges = []
        remove_edges = set()
        remove_nodes = set()
        for match in matches:
            op, *args = match
            add = op == "add"
            if add or (op == "delete" and len(args)==2):
                a, b = args
                if a == b:
                    continue
                if add:
                    new_edges.append((a, b))
                else:
                    # remove both directions
                    # (undirected graph)
                    remove_edges.add(frozenset([a, b]))
            else: # must be delete of node
                remove_nodes.add(args[0])

        if replace:
            edges = new_edges
        else:
            edges = self.edges + new_edges

        # make sure edges aren't added twice
        # and remove nodes/edges that were deleted
        added = set()
        for edge in edges:
            nodes = frozenset(edge)
            if nodes in added or nodes & remove_nodes or nodes in remove_edges:
                continue
            added.add(nodes)

        self.edges = list([tuple(a) for a in added])
        self.nodes = list(set([n for e in self.edges for n in e]))
        self.save()

    def _delete_node(self, node) -> None:
        """Delete a node and all edges connected to it.

        Args:
            node (str): The node to delete.
        """
        self.edges = [e for e in self.edges if node not in frozenset(e)]
        self.nodes = list(set([n for e in self.edges for n in e]))
        self.conversation.append(Message(
            f'delete("{node}")', 
            role="user"
        ))
        self.save()

    def _add_expand_delete_buttons(self, node) -> None:
            st.subheader(node)
            # Create a container for the buttons


            # Display the buttons in the container
            st.button(
            label="Expand", 
            on_click=self.ask_for_extended_graph,
            key=f"expand_{node}",
            # pass to on_click (self.ask_for_extended_graph)
            kwargs={"selected_node": node}
            )
            st.button(
            label="Delete", 
            on_click=self._delete_node,
            type="primary",
            key=f"delete_{node}",
            # pass on to _delete_node
            args=(node,)
            )
# ======================================================== Visualization =============================================================================
    def visualize(self, graph_type: Literal["agraph", "networkx", "graphviz"]) -> None:
        """Visualize the mindmap as a graph a certain way depending on the `graph_type`.

        Args:
            graph_type (Literal["agraph", "networkx", "graphviz"]): The graph type to visualize the mindmap as.
        Returns:
            Union[str, None]: Any output from the clicking the graph or 
                if selecting a node in the sidebar.
        """

        selected = st.session_state.get("last_expanded")
        if graph_type == "agraph":
            vis_nodes = [
                Node(
                    id=n, 
                    label=n, 
                    # a little bit bigger if selected
                    size=10+10*(n==selected), 
                    # a different color if selected
                    color=COLOR if n != selected else FOCUS_COLOR,
                ) 
                for n in self.nodes
            ]
            vis_edges = [Edge(source=a, target=b, color=EDGE_COLOR) for a, b in self.edges]
            config = Config(width="100%",
                            height=400,
                            directed=False, 
                            physics=True,
                            hierarchical=False,
                            )
            # returns a node if clicked, otherwise None
            clicked_node = agraph(nodes=vis_nodes, 
                            edges=vis_edges, 
                            config=config)
            # if clicked, update the sidebar with a button to create it
            
            if clicked_node is not None:
                self._add_expand_delete_buttons(clicked_node)
                return
#===================================================================== Main ===============================================================
def main():
    # will initialize the graph from session state
    # (if it exists) otherwise will create a new one
    
    mindmap = MindMap.load()
    
    st.markdown("""
    <style>
    .stButton button {
        background-color: #b9359a;  /* Change to your desired color */
        color: white;
        border-color: transparent;  /* Set the border color to transparent */
    }

    .stButton button:hover {
        background-color: #ffffff;  /* Retain the same color when hovering */
        border-color:#b9359a ;  /* Set the border color to transparent */
        color:  #b9359a;

     .stTextArea > div > textarea {
        border-color: #6dbbbd; /* Change to your desired color */
        background-color: #6dbbbd;  /* Retain the same color when hovering */
        color: white;
    .header {
    font-size: 20px;  /* Set the desired font size */
    }
    .button-container {
    display: flex;
    }
    }
    }
    }
    </style>
    """, unsafe_allow_html=True)

    with col1:
        st.image("https://raw.githubusercontent.com/AnasMations/StudySync/main/img/Study%20Sync.png", width=200)
        st.markdown("<h4 class='header'>AI Mind Map Generator</h4>", unsafe_allow_html=True)
      
        empty = mindmap.is_empty()
        reset = empty or st.checkbox("Reset mind map", value=False)
        query = st.text_area(
            "**Describe your mind map**" if reset else "**Describe how to change your mind map**", 
            value=st.session_state.get("mindmap-input", ""),
            key="mindmap-input",
            height=150,
            placeholder="Generate a mindmap for sorting algorithms in computer science",
        )
        submit = st.button("Submit")

    with col2:
        graph_type = "agraph"
    
        valid_submission = submit and query != ""

        if empty and not valid_submission:
            return

        with st.spinner(text="Loading graph..."):
            # if submit and non-empty query, then update graph
            if valid_submission:
                if reset:
                    # completely new mindmap
                    mindmap.ask_for_initial_graph(query=query)
                else:
                    # extend existing mindmap
                    mindmap.ask_for_extended_graph(text=query)
                # since inputs also have to be updated, everything
                # is rerun
                st.experimental_rerun()
            else:
                mindmap.visualize(graph_type)

  

if __name__ == "__main__":
    main()