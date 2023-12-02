# Standard Libraries
import json
import os
import requests

# Suppressing warnings from urllib3
import urllib3
urllib3.disable_warnings()

# .env for environment variables
from dotenv import load_dotenv

# Imports related to LangChain
from langchain import LLMChain, PromptTemplate
from langchain.agents import AgentExecutor, AgentType, Tool, initialize_agent, load_tools, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Other utilities and types
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Type

# Imports related to Splunk
import splunklib.client as client
import splunklib.results as results
from splunklib.binding import HTTPError

# Streamlit for web app interface
import streamlit as st

# Local import for prompts
from prompts import *

# Load environment variables
load_dotenv()

serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
browserless_api_key = os.getenv('BROWSERLESS_API_KEY')
splunk_url = os.getenv('SPLUNK_URL')
splunk_username = os.getenv('SPLUNK_USERNAME')
splunk_password = os.getenv('SPLUNK_PASSWORD')

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.0)
llm4 = ChatOpenAI(model_name="gpt-4", temperature=0.0)

#model_id = 'meta-llama/Llama-2-7b-chat-hf'

#
# Chains
#
start_chain = LLMChain(llm=llm, prompt=tasks_initializer_prompt, verbose=False)
detial_chain = LLMChain(llm=llm, prompt=tasks_details_agent, verbose=False)
tasks_context_chain = LLMChain(llm=llm, prompt=tasks_context_agent, verbose=False)
tasks_human_chain = LLMChain(llm=llm, prompt=tasks_human_agent, verbose=False)
task_assigner_chain = LLMChain(llm=llm, prompt=task_assigner_agent, verbose=False)
spl_writer_chain = LLMChain(llm=llm, prompt=spl_writer_agent, verbose=False)
spl_refactor_chain = LLMChain(llm=llm, prompt=spl_refactor_agent, verbose=False)
event_id_chain = LLMChain(llm=llm, prompt=event_id_prompt, verbose=False)
spl_normalize_chain = LLMChain(llm=llm, prompt=spl_normalize_agent, verbose=False)
spl_summary_chain = LLMChain(llm=llm, prompt=summarize_splunk_results, verbose=False)
spl_writer_agent_testing_chain = LLMChain(llm=llm, prompt=spl_writer_agent_testing, verbose=False)
tasks_details_agent_testing_chain = LLMChain(llm=llm, prompt=tasks_details_agent_testing, verbose=False)
spl_filter_agent_chain = LLMChain(llm=llm, prompt=spl_filter_agent, verbose=False)
spl_statistical_analysis_chain = LLMChain(llm=llm, prompt=spl_statistical_analysis_agent, verbose=False)
spl_statistical_analysis_chain = LLMChain(llm=llm, prompt=spl_statistical_analysis_agent, verbose=False)
splunk_human_input_agent_chain = LLMChain(llm=llm, prompt=splunk_human_input_agent, verbose=False)

# End Chains

#
# Helper Functions/Classes
#
class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")
class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput
    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)
    def _arun(self, url: str):
        raise NotImplementedError("error here")
def search(query):
    '''
    Purpose:

    returns: list of
    '''
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    #print(response.text)
    return response.text
def scrape_website(objective: str, url: str):
    '''
    scrape website, and also will summarize the content based on objective if the content is too large
    objective is the original objective & task that user give to the agent, url is the url of the website to be scraped
    '''
    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    # Define the data to be sent in the request
    data = {"url": url}
    # Convert Python object to JSON string
    data_json = json.dumps(data)
    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        output = summary(objective, text)
        return output
        # if len(text) > 10000:
        #     output = summary(objective, text)
        #     return output
        # else:
        #     return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")
def summary(objective, content):
    '''
    Purpose:

    returns:
    '''
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}. It is important that you include relevant Windows Event ID, Field Names, expected values for given fields.
    These will be important when using the summary as context to build a Splunk SPL detection query.

    TEXT:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    summary_chain = load_summarize_chain( llm=llm, chain_type='map_reduce', map_prompt=map_prompt_template, combine_prompt=map_prompt_template, verbose=True)
    output = summary_chain.run(input_documents=docs, objective=objective)
    return output

def run_splunk_search(search_query: str) -> list:
    """
    Run a Splunk search and return the results.

    Parameters:
    - search_query (str): Splunk search query.

    Returns:
    - list: List of search results.
    """
    service = client.connect(
        host=splunk_url,
        username=splunk_username,
        password=splunk_password,
        autologin=True
    )

    kwargs_export = {
        "earliest": '-7d',
        "latest_time": "now",
        "output_mode": 'json'
    }

    try:
        if not search_query.strip().lower().startswith("search"):
            search_query = "search " + search_query
        job = service.jobs.create(search_query)
        oneshot_results = service.jobs.oneshot(search_query, **kwargs_export)
        content = results.JSONResultsReader(oneshot_results)
        return content

    except HTTPError as e:
        error_message = str(e)
        error_portion = error_message.split("Error at position", 1)
        if len(error_portion) > 1:
            return f"Error at position {error_portion[1]}"
        else:
            print(f"Error: {e}")
            return e


#
# Loop Handlers
#

def handle_splunk_executor_agent(task, spl_command):
    results_list = [item for item in run_splunk_search(spl_command)]
    return results_list

def handle_spl_writer_agent(task, objective, schema, splunk_info):
    st.markdown("<span style='color: blue;'>Writing Some SPL ...</span>", unsafe_allow_html=True)
    return spl_writer_chain.predict(objective=objective, task=task["description"], isolated_context=task["isolated_context"], splunk_info=splunk_info,schema=schema)

def handle_spl_filter_agent(task, objective, spl_command):
    st.markdown("<span style='color: blue;'>Applying SPL Filters ...</span>", unsafe_allow_html=True)
    return spl_filter_agent_chain.predict(objective=objective, task=task["description"], previous_query=spl_command, isolated_context=task["isolated_context"])

def handle_spl_statistical_analysis_agent(task, objective, spl_command):
    st.markdown("<span style='color: blue;'>Applying SPL Statistical Analysis ...</span>", unsafe_allow_html=True)
    return spl_statistical_analysis_chain.predict(objective=objective, task=task["description"], previous_query=spl_command, isolated_context=task["isolated_context"])

def handle_spl_refactor_agent(task, objective, spl_command, splunk_info, schema):
    st.markdown("<span style='color: blue;'>Refactoring SPL ...</span>", unsafe_allow_html=True)
    return spl_normalize_chain.predict(existing_spl=spl_command, objective=objective, splunk_info=splunk_info, schema=schema)

def handle_spl_results_agent(objective, query, splunk_results):
    final_data=[]
    for item in splunk_results:
        #print(item)
        final_data.append(item)
    return spl_summary_chain.predict(objective=objective, query=query, results=final_data)
    

### END HELPER ###


### Start TOOLS ###
loader = TextLoader('./content/BlogPostSplunkGPT.txt')
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
docsearch = FAISS.from_documents(docs, embeddings)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
research_tools = [
Tool(
    name="Internet_Search",
    func=search,
    description="Internet Search: useful for when you need to answer questions about current events, internet data. You should ask targeted questions"
),
ScrapeWebsiteTool(),
Tool(
    name="Local_Search",
    func=qa.run,
    description="Local Search: useful for when you need to answer questions about current events, using local data. You should ask targeted questions",
),]

### END TOOLS ###