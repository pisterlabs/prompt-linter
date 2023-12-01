#plotly_chart.py
import streamlit as st
import numpy as np
from random import randrange
import openai,boto3,urllib, requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import re, time,os
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
from serpapi import GoogleSearch

from graphviz import Digraph
import ast
import urllib.request
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from bs4.element import Comment

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

def serp_news_search (query_json) :

	search = query_json['query']
	number = query_json['limit']

	params = {
        "api_key": st.secrets["SERP_API_KEY"],         # https://serpapi.com/manage-api-key
        "engine": "google",       # serpapi parsing engine
        "q": search,         # search query
        "tbm": "nws"  ,            # news results
        "h1": "en",                # language
        "gl": "us",                # country to search from
        "google_domain": "google.com",
        "num": number,
        "tbs": "qdr:d3" # last week
        }


	search = GoogleSearch(params) # where data extraction happens
	results = search.get_dict()   # returns an iterator of all pages
	news_dict_arr = []
	for result in results["news_results"]:
		news_dict_arr.append(result)
	#titles_and_links = [ {'title': news_dict_arr[i]['title'], 'link': news_dict_arr[i]['link'], 'snippet': news_dict_arr[i]['snippet']}   for i in range(len(news_dict_arr)) ]
	titles_and_links = [ {'title': news_dict_arr[i]['title'], 'link': news_dict_arr[i]['link'] }   for i in range(len(news_dict_arr)) ]
	
	# Extracting links from titles_and_links
	links = [item['link'] for item in titles_and_links if 'youtube' not in item['link'] ]

 	#links = [ link_dict['link'] for link_dict in titles_and_links if 'youtube' not in link ]
	
	return links

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

st.set_page_config(page_title="Auto Research and Report",layout='wide')

st.header("Auto Research - Auto Web Research and Write Professional Looking Report on Any Topic")

#url = 'https://investrecipes.s3.amazonaws.com/knowledge_graph.gv'

#import urllib.request
#with urllib.request.urlopen(url) as f:
    #html = f.read().decode('utf-8')

#st.graphviz_chart(html)
url = "https://thehackernews.com/2023/09/financially-motivated-unc3944-threat.html"
url = "https://nvd.nist.gov/vuln/detail/CVE-2023-41331"

button_name = "Write Report"
response_while = "Right on it,  searching the web  .. it should be around 5-10 seconds ... (research limited to top 2 links)"
response_after = "Here you go ...  "

#partial_url = st.text_input('Enter any URL (ex - https://msrc.microsoft.com/blog/2023/09/results-of-major-technical-investigations-for-storm-0558-key-acquisition/) or a CVE ID (ex - CVE-2023-35708)', 'CVE-2023-35708')
objective = st.text_input("Topic you want to research and write a report on (ex- 'report on latest CURL vulnerabilities' or 'What are the stocks to buy in war times ?' )", 'Report on latest CURL vulnerabilities')

links = serp_news_search({'query': objective, 'limit': 2})

current_report = "NA"
sec_q_button=st.button(button_name, key = 'sec_q_button')
st.markdown ( "--------")
if sec_q_button :
    with st.spinner ( response_while ) :
        with st.empty ():
            
            for link in links :
                st.write ('reading ...  ' + link)
                #st.markdown('#### Reading the URL ' + link)
                multi = ''' 
                     __Reading ...__    {link}
                     
                     
                     **Analyzing and writing the report**
                '''.format(link=link)
                #st.markdown (' __Reading ...__   ' + link + ' \n **Analyzing and writing the report** ')
                st.markdown (multi)
                request = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})

                #html = urllib.request.urlopen(sys.argv[1]).read()
                #print(text_from_html(html))
                #initiate_driver_return_browser(sys.argv[1])
                webpage = urlopen(request).read()
                text = text_from_html (webpage)
                #print (text)
                system_content = """ 
                You are an excellent analyst with the ability to write comprehensive research report based on the information given to you.
                """
                prompt_content = """ 

                    # Your Role
                            You are an excellent analyst with the ability to write comprehensive research report based on the 
                            information given to you.
                    # Objective
                    {objective}
                    # Your Task


                    Your job is to write a comprehensive report to fulfill the primary goals
                    under OBJECTIVE, using the information given to you.
                    To accomplish this task, you would be given three different sets of information. 
                    - The first set of information you will be given is the data collected from a website.
                    - The second set of information is an existing report that was written from data sourced from different websites and corresponding URLs(s).
                    - The third set of information is the URL of the website itself from which the data is soruced.
                    ### Location of the information that I am passing to you
                    - The data that is given to you is sourced from a website, that is under ## New_URL Section.  
                    - The data that is sourced that you will be basing the report on is under te # New_Data Section.
                    - The existing report that is given to you is under the # Current_Report section.
                    ### Contexts of writing the report
                    There can be two different contexts under which you have to write the report 
                    - There is a current report that you have to enhance : if the # Current_Report section is not 'NA', then your task is to enhance the report in the # Current_Report section  with the information that I am supplying is in the newdata section.  If that is the case, add the new URL that is in the New_URL section to the reference URL table, and enhance the content in the current report, add the citations for the new URL and produce the new report.
                    - There is no existing report, you are starting fresh: If the # Current_Report Section is "NA", then, your task is to write the report for the objective afresh, since there is no existing report.

                    # Your Response and Output Format
                    You should ONLY respond in the Markdown format as described below
                    Markdown document with a title (which is the objective) and multiple sections, each with up to 2000 words.
                    Start each section with a H1 heading with emoji (e.g. `# ⛳️ Title`).  use approritate emoji for each section's and subection heading and content.  Add references to the URLs in the text of the report using the [^number^] format, where number is the id of the URL reference so the reader can reference the sources of the text in your report.
                    
                    You must keep the URLs and your notes in the  Notes_And_References section, which must be the last section of the report

                    Instructions on how to build up the Notes_And_References section -
                    1. It has two subsections, ##Notes and ##Reference_URLs.

                    ### Instructions on how to write notes section 
                    1. The ##Notes must have notes around what you changed in the existing report and why.  
                    #### Instructions on how to build the #Reference_URLs section is as follows:
                    1. The "Reference_URLs" section must have a table, with 5 columns: Citation ID, URL and Summary, "New or Existing URL" and "Sections where References are Added". 
                    1. You must include the URL (or add to the existing table) in the Reference_URL section.
                    1. You must add the URL Citation ID and corresponding citations to the report as you are enhancing the current report or writing a fresh report

                    # Check list Before your show me the report 

                    Think step by step, and produce the new report.  Before producing the report, check the following and add how you accomplished these to the notes section -

                            - check 1 : How many existing URLs are there in the current report ? Did you keep all the existing and relevant URLs and corresponding citations in the old Reference_URL table and not remove them ? --> this is a must pls check
                            - check 2 : Did you add the new URL supplied to the reference table using the [^num^] notation ? what is the new url and did you add the fact that its a new URL to the table ? what is the ciation ID for the new URL ? which sections did you add the citations with that citation ID of the new URL ? list the sections 
                        
                            - Give importance to numbers such as dates, percentages, and other numerical data to include in the report
                # New_URL
                {url}

                # New_Data
                {new_data}

                # Current_Report
                {current_report}
                """

                prompt_string =prompt_content.format(objective = objective, url=link, new_data=text, current_report=current_report) 
                completion = chat_complete (model = "gpt-3.5-turbo-16k", system_content=system_content, temperature=0.2, user_content=prompt_string).completion


                st.markdown (completion['choices'][0]["message"]["content"])
                current_report = completion['choices'][0]["message"]["content"]
                time.sleep (7)