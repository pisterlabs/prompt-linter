import shodan
import re
from langchain.tools import BaseTool, Tool
from typing import Optional
from llms.azure_llms import create_llm
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate, LLMChain

tool_llm = create_llm(temp=0.4)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# your shodan API key
SHODAN_API_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
api = shodan.Shodan(SHODAN_API_KEY)

def shodan_ip_search(ip):
    result = ""
    try:
        # Lookup the host
        host = api.host(ip)
        # Build general info string
        general_info = """
            IP: {}
            Organization: {}
            Operating System: {}
        """.format(host['ip_str'], host.get('org', 'n/a'), host.get('os', 'n/a'))
        # Append general info to result
        result += general_info
        # Build banners string
        banners = ""
        for item in host['data']:
            banner = """
                Port: {}
                Banner: {}

            """.format(item['port'], item['data'])
            banners += banner
        # Append banners to result
        result += banners

    except shodan.APIError as e:
        result = 'Error: {}'.format(e)

    return result

class shodan_ip_lookup_tool(BaseTool):
    name = "Shodan IP Lookup"
    description = "use Shodan to get info on any exposed services and potential vulnerabilities associated with a IP. Input is an ip address."
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        try:
            ip = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', query).group()
            response = shodan_ip_search(ip)
            prompt = "User: Analyze above data and report on exposed services and potential vulnerabilities"
            return (response+prompt)
        except:
            return "Shodan ip host search tool not available for use."

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
ip_lookup = shodan_ip_lookup_tool()
shodan_ip_lookup_tool = Tool(
    name = "Shodan IP Lookup",
    description = "use Shodan to get info on any exposed services and potential vulnerabilities associated with a IP. Input is an ip address.",
    func= ip_lookup.run
    )
