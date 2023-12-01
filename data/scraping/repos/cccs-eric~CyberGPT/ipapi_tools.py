import re
import requests
# Import things that are needed generically
from langchain.tools import BaseTool, Tool
from typing import Optional
from llms.azure_llms import create_llm
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate, LLMChain
from tools.borealis_tools import extract_ips_urls_domains

tool_llm = create_llm(temp=0.4)
tool_llm_temp0 = create_llm(temp=0)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

def make_request(url):
    try:
        req = requests.get(url)
        return req.text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        return "IP-API request failed for : "+str(url)
            
def get_ipapi_response(query_list):
    response = []
    for elem in query_list:
        url = 'http://ip-api.com/json/'+elem
        answer = make_request(url)
        response.append(answer)
    return response

def ipapi_processing(query):
    ips, urls, domains = extract_ips_urls_domains(query)
    if len(ips) > 0 or len(domains) > 0:
        query_list = ips+domains
        response = get_ipapi_response(query_list)
        result = '\n'.join(response)
        return (result)
    else:
        return None

class ipapi_tool(BaseTool):
    name = "IP Lookup"
    description = "use for getting an ip address from a domain, as well as geolocation and internet provider information"
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        try:
            return ipapi_processing(query)
        except Exception as e:
            return str(e)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
ipapi_lookup = ipapi_tool()
ipapi_tool = Tool(
    name = "IP Lookup",
    description = "use for getting an ip address from a domain, as well as geolocation and internet provider information",
    func= ipapi_lookup.run
    )
