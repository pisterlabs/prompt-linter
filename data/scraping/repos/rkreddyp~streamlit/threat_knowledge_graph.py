#plotly_chart.py
import streamlit as st
import numpy as np
from random import randrange
import openai,boto3,urllib, requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import re, time
from pydantic import BaseModel, Field
from typing import List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile


from graphviz import Digraph
import ast

os.environ['OPENAI_API_KEY'] = st.secrets['OPEN_API_KEY']
    
def openai_schema(cls):
    schema = cls.schema()
    parameters = {
        k: v for k, v in schema.items() if k not in ("title", "description")
    }
    parameters["required"] = sorted(parameters["properties"])
    
    return {
        "name": schema["title"],
        #"description": schema["description"],
        "parameters": parameters,
    }

## main
class chat_complete:

    def __init__(self, user_content, model="gpt-3.5-turbo-16k", temperature=0.1, functions=None, system_content="You are a helpful assistant"):
        self.model = model
        self.temperature = temperature
        self.functions = functions if functions is not None else []
        self.messages = [{"role": "user", "content": user_content}]
        if system_content:
            self.messages.insert(0, {"role": "system", "content": system_content})

        self.completion = self.call_openai_create()
        
    
    def call_openai_create(self):
        task = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": self.messages
        }

        # Only add 'functions' key if self.functions is not empty
        if self.functions:
            task['functions'] = self.functions
            task['function_call'] = {"name": self.functions[0]["name"]}

        # Here we use OpenAI's API to create a completion
        # completion = openai.ChatCompletion.create(**task)  # As per your original code
        completion = openai.ChatCompletion.create(**task)  # As per OpenAI's API as of September 2021
        # dlogging.info (completion)
        return completion.to_dict()

def initiate_driver_return_browser(url):
    print ( "initiate_driver_return_browser")
    opts = FirefoxOptions()

    fp = webdriver.FirefoxProfile()
    fp.set_preference("security.fileuri.strict_origin_policy", False);
    fp.set_preference("javascript.enabled", False);
    fp.update_preferences()
    opts.add_argument("--headless")
    opts.set_preference("browser.download.folderList", 2)
    opts.set_preference("browser.download.dir", "/tmp/") 
    opts.set_preference("browser.helperApps.neverAsk.saveToDisk","application/text/csv")
    #browser = webdriver.Firefox(firefox_options=opts , log_path='/tmp/geckodriver.log', executable_path = '/tmp/geckodriver', firefox_profile=fp)
    try :
        #browser = webdriver.Firefox(options=opts , log_path='/tmp/geckodriver.log', executable_path = '/tmp/geckodriver')
        browser = webdriver.Firefox(options=opts)
    except :
        time.sleep(5)
        try :
            #browser = webdriver.Firefox(options=opts , log_path='/tmp/geckodriver.log', executable_path = '/tmp/geckodriver')
            browser = webdriver.Firefox(options=opts )
        except Exception as e: print(e)

    delay = 4
    browser.set_window_size(1920,1920)
    try :
        browser.get(url)
    except Exception as e:
        print(e) 
    
    time.sleep(2)
    return browser 

def fetch_text_requests(url) -> List[str]:
        import requests
        from bs4 import BeautifulSoup
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        body = soup.find('body').get_text()
        return body.split('\n')
    

def fetch_text(url) -> List[str]:
        text_arr = []
        try :

            browser = initiate_driver_return_browser(url)
            el = browser.find_element(By.TAG_NAME,'body')
            for text in (el.text).split('\n'):
                if len (text) > 200:
                    #print (url)
                    #print ("fetchtext --", text)
                    text_arr.append(text)
            return ".".join (text_arr)
        except Exception as e:
            st.write (e)
            print ("exceptin in fetch_text")
            return "NA"

class Node(BaseModel):
    """
    Node class for the knowledge graph. Each node represents an entity.
    
    Attributes:
        id (int): Unique identifier for the node.
        label (str): Label or name of the node.
        color (str): Color of the node.
        num_targets (int): Number of target nodes connected to this node.
        num_sources (int): Number of source nodes this node is connected from.
        list_target_ids (List[int]): List of unique identifiers of target nodes for which this node is the source node.
        num_edges (int): Total number of edges that this node is a part of, either soruce or target.
        
    """
    
    id: int
    label: str
    color: str
    num_targets: int
    num_sources: int
    list_target_ids: List[int] = Field(default_factory=list)
    num_edges: int = 0

class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

def visualize_knowledge_graph(kg):
    print (kg.keys())
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg['nodes']:
        dot.node(str(node['id']), node['label'], color=node['color'])

    # Add edges
    for edge in kg['edges']:
        #dot.edge(str(edge['source']), str(edge['target']), label=edge['label'], color=edge['color'])
        dot.edge(str(edge['source']), str(edge['target']), label=edge['label'])

    # Render the graph
    dot.render("/tmp/knowledge_graph.gv", view=False)

st.set_page_config(page_title="SecurityGPT Threat Knowledge Graphs",layout='wide')

st.header("SecurityGPT - Draw any URL as a Threat Knowledge Graphs")

#url = 'https://investrecipes.s3.amazonaws.com/knowledge_graph.gv'

#import urllib.request
#with urllib.request.urlopen(url) as f:
    #html = f.read().decode('utf-8')

#st.graphviz_chart(html)
url = "https://thehackernews.com/2023/09/financially-motivated-unc3944-threat.html"
url = "https://nvd.nist.gov/vuln/detail/CVE-2023-41331"

button_name = "Draw Knowledge Graph"
response_while = "Right on it, it should be around 5-10 seconds ..."
response_after = "Here you go ...  "

partial_url = st.text_input('Enter any URL (ex - https://msrc.microsoft.com/blog/2023/09/results-of-major-technical-investigations-for-storm-0558-key-acquisition/) or a CVE ID (ex - CVE-2023-35708)', 'CVE-2023-35708')
objective = st.text_input("objective", 'understand the technical details cyber security attack exhaustively, the who (who is responsible, who got affected), the what (techniques), how it started, how it propagated')

sec_q_button=st.button(button_name, key = 'sec_q_button')
st.markdown ( "--------")
if sec_q_button :
    with st.spinner ( response_while ) :

        if 'cve' in partial_url.lower():
            url = "https://nvd.nist.gov/vuln/detail/" + partial_url
        else :
            url= partial_url
        #text = fetch_text(url)
        #st.write(text)
        #st.write(url)
        st.write ('fetching the url - ' + url )
        kc= fetch_text_requests(url)
        objective = objective
        kg_schema = openai_schema (KnowledgeGraph)
        st.write ('parsed the url, sending to openai for graph generation....')
        system_content = "You are an an awesome information security engineer and detailed knowledge graph developer"

        prompt_content = """ 
        Your task is make the knowledge graph from an article text for a given objective.
        the article test is under the section # article_text
        objective : {objective}
        you must follow all reqirements that are listed below
        
        ## graph requirements
        - in the edges list, count the number of target ids for each source ids, and the
        number must be list_target_ids for each node
        - when you make the edges, ensure that there must not be more than 10 target node 
            ids connected to any given source node id in the graph. the number in list_target_ids
            for any given source node id must therefore be at most 10
        - for edge colors, you must make it visually informative.  you must use red color for 
            important bad items , green color for good items, be creative for other colors
        - there graph should be uni-directional, no bi-directional edges.  for example,
            if node 1 is connected to node 2, then node 2 should not be connected to node 1
        - the number of target nodes that are attached to source node of id 1 should be at most 5
        - group all related targets belonging to a theme to one soruce id
        - all nodes in the graph should be connected to atleast one other node
        - all labels should be text and strings, not integers
        - num_edges for all nodes should be 2 or more
        - all nodes should be connected to node with source id 1 , directly or indirectly
        - focus on specific information such time periods, dates, numbers , percentages, make them as edges such as "on" or "during"
        - if time periods are mentioned, use them as edges and put the nodes and IDs in chronological order
        - ensure to be as exhaustive as you can, include all the details from the article to acheive the objective
        
        # example of a good graph output
        ```
            {{'nodes': [{{'id': 1,
            'label': 'Attack Details (SOFARPC)',
            'color': 'blue',
            'num_targets': 4,
            'num_sources': 0,
            'list_target_ids': [3,4,8,18],
            'num_edges': 4}},
            {{'id': 2,
            'label': 'Java RPC framework',
            'color': 'blue',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [1],
            'num_edges': 2}},
            {{'id': 3,
            'label': 'Versions prior to 5.11.0',
            'color': 'blue',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [1],
            'num_edges': 2}},
            {{'id': 4,
            'label': 'Remote command execution',
            'color': 'red',
            'num_targets': 1,
            'num_sources': 2,
            'list_target_ids': [1,5],
            'num_edges': 3}},
            {{'id': 5,
            'label': 'Payload',
            'color': 'red',
            'num_targets': 2,
            'num_sources': 1,
            'list_target_ids': [4,6,7],
            'num_edges': 3}},
            {{'id': 6,
            'label': 'JNDI injection',
            'color': 'red',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [5],
            'num_edges': 2}},
            {{'id': 7,
            'label': 'System command execution',
            'color': 'red',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [5],
            'num_edges': 2}},
            {{'id': 9,
            'label': 'Version 5.11.0',
            'color': 'green',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [8],
            'num_edges': 2}},
            {{'id': 18,
            'label': 'Fix',
            'color': 'green',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [1],
            'num_edges': 2}},
            {{'id': 8,
            'label': 'Workarounds and Fixe Details',
            'color': 'green',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [9],
            'num_edges': 2}},
            {{'id': 11,
            'label': '-Drpc_serialize_blacklist_override=javax.sound.sampled.AudioFileFormat',
            'color': 'green',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [18],
            'num_edges': 2}},
            ],
        'edges': [{{'source': 1, 'target': 2, 'label': 'is a', 'color': 'blue'}},
            {{'source': 1, 'target': 3, 'label': 'Versions prior to', 'color': 'blue'}},
            {{'source': 1, 'target': 4, 'label': 'vulnerable to', 'color': 'red'}},
            {{'source': 4, 'target': 5, 'label': 'can achieve', 'color': 'blue'}},
            {{'source': 5, 'target': 6, 'label': 'or', 'color': 'red'}},
            {{'source': 5, 'target': 7, 'label': 'or', 'color': 'red'}},
            {{'source': 1, 'target': 8, 'label': 'work around', 'color': 'red'}},
            {{'source': 8, 'target': 9, 'label': 'or', 'color': 'green'}},
            {{'source': 1, 'target': 18, 'label': 'fixed by', 'color': 'red'}},
            {{'source': 18, 'target': 11, 'label': 'or', 'color': 'green'}},
            
            ]}}
        ```
        
        think step by step, 
        before you give the graph to the user =
        - are the list_target_ids filled up for all nodes ?
        - is the number of list_target_ids (num_targets) more than 8 ?
        - do not hallucinate, do not repeat the example, analyze the text and make the graph
        then re-do the graph

        the attack article is as follows : 
        # article_text
        """

        prompt_string =prompt_content.format(objective = objective) + "\n" + str(kc)
        completion = chat_complete (model = "gpt-3.5-turbo-16k", system_content=system_content, temperature=0.2, user_content=prompt_string, functions = [kg_schema] ).completion

        #st.write (completion)
        visualize_knowledge_graph ( ast.literal_eval (completion['choices'][0].message['function_call']['arguments']) )

        with open('/tmp/knowledge_graph.gv', 'r') as file:
            file_contents = file.read()

        st.graphviz_chart (file_contents)
