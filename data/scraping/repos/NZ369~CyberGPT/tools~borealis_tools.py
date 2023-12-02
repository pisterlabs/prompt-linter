import re
import requests
# Import things that are needed generically
from langchain.tools import BaseTool, Tool
from typing import Optional
from llms.azure_llms import create_llm
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate, LLMChain

tool_llm = create_llm(temp=0.4)
tool_llm_temp0 = create_llm(temp=0)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# your shodan API key
BOREALIS_KEY = 'af0b5bd9bdee4c4db548711a326bab23'

info_template = """Create a analysis report using this data, include all links generated by modules in a References section: {info}"""
info_prompt_template = PromptTemplate(input_variables=["info"], template=info_template)
answer_chain = LLMChain(llm=tool_llm, prompt=info_prompt_template)

def extract_ips_urls_domains(text):
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    domain_pattern = r'(?:[-\w]+\.)+[a-zA-Z]{2,}'

    ips = re.findall(ip_pattern, text)
    urls = re.findall(url_pattern, text)
    domains = re.findall(domain_pattern, text)

    return ips, urls, domains

def make_request(url):
    try:
        req = requests.get(url)
        return req.text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        return "Borealis request failed for : "+str(url)
            
def get_borealis_response(query_list, type="all"):
    response = []
    modules = 'BEAVER,VIRUSTOTAL,MOOSE,STONEWALL,AUWL,SAFEBROWSING,ALPHABETSOUP,MAXMIND'
    for elem in query_list:
        if type == "all":
            url = 'https://ingestion.collaboration.cyber.gc.ca/borealis/process/'+ elem +'?modules='+ modules +'&subscription-key=' + BOREALIS_KEY
            answer = make_request(url)
            response.append(answer_chain.run(answer))
        else:
            print("Invalid type specified. Please provide 'all' as the type.")
    return response

def borealis_processing(query):
    ips, urls, domains = extract_ips_urls_domains(query)
    if len(ips) > 0 or len(domains) > 0:
        query_list = ips+domains
        response = get_borealis_response(query_list)
        result = '\n'.join(response)
        return (result)
    else:
        return None

class borealis_tool(BaseTool):
    name = "Borealis IP URL Domain Lookup"
    description = "Use Borealis for getting information about ip, url and domains from multiple threat vetting tools"
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        try:
            return borealis_processing(query)
        except Exception as e:
            return str(e)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
borealis_lookup = borealis_tool()
borealis_tool = Tool(
    name = "Borealis IP URL Domain Lookup",
    description = "Use Borealis for getting information about ip, url and domains from multiple threat vetting tools",
    func= borealis_lookup.run
    )