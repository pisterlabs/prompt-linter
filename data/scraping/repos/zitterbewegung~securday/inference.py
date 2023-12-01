import ipaddress, os, re, socket, json
import urllib.request
import urllib.parse
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from shodan import Shodan

import pwnlib
from langchain.chat_models import ChatOpenAI

# import cve_searchsploit as CS
from langchain.docstore import Wikipedia

from langchain.llms import OpenAI, PromptLayerOpenAI
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.agents import Tool, AgentType, tool
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent
_executor, load_chat_planner
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import load_tools, initialize_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents import Tool, AgentExecutor
from langchain_experimental.utilities import PythonREPL
from langchain.tools.ifttt import IFTTTWebhook
from langchain.utilities import (
    WikipediaAPIWrapper,
    TextRequestsWrapper,
)
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.tools import ShellTool
from censys.search import CensysHosts
import vt
load_dotenv()

memory = ConversationBufferMemory()
wolfram = WolframAlphaAPIWrapper()
wikipedia = WikipediaAPIWrapper()
python_repl =  PythonREPLTool()
requests = TextRequestsWrapper()
shodan_api = Shodan(os.environ.get("SHODAN_API_KEY"))
virus_total_client = vt.Client(os.environ.get("VIRUS_TOTAL"))
shell_tool = ShellTool()
censys_hosts = CensysHosts()
set_llm_cache(InMemoryCache())


def hostname(hostname: str) -> str:
    """useful when you need to get the ipaddress associated with a hostname"""
    try:
        ip = ipaddress.ip_address(addr)
        return ip
    except ValueError:
        return "Invalid ip address"


def subset_shodan(addr: str):
    # ipv4_extract_pattern = "(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.(?:25[0
-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.(
?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    # extracted_ip = re.findall(ipv4_extract_pattern, addr)[0]
    # if ipaddress.ip_address(extracted_ip).is_private:
    #    return "This is a private ip address."
    addr = addr.replace("scan ", "")
    try:
        host = shodan_api.host(addr)
    except Exception as e:
        return "Shodan has no info"
    ports = ""
    for i in host["data"]:
        ports += "Port {} \n".format(i["port"])

    return """
    IP: {}
    Organization: {}
    Operating System: {}
    Country: {}
    Location: Lat {} Long {}
    Asn: {}
    Transport: {}
    Port: {}
    """.format(
        host["ip_str"],
        host.get("org", "n/a"),
        host.get("os", "n/a"),
        host.get("country_name", "n/a"),
        host.get("lat", "n/a"),
        host.get("long", "n/a"),
        host.get("asn", "n/a"),
        host.get("transport", "n/a"),
        ports,
    )



def censys_find_location(addr: str):
    host_query = censys_hosts.view(addr)
    location_dictionary = host_query.get('location')
    
    return location_dictionary

def shell_wrapper(query: str):
    return shell_tool.run({"commands": [query]})
    
def virus_total(url):
    """Takes a URL and aggregates the result of malware on the site."""
    url_id = vt.url_id(url)
    url = virus_total_client.get_object("/urls/{}", url_id)
    
    analysis = url.last_analysis_stats
    
    return """File fetched from URL is
    harmess {},
    malicious {},
    suspicious {}
    . """.format(analysis.get('harmless'),
		 analysis.get("malicious"),
		 analysis.get("suspicious"),)

def scan_ip_addr(ipaddress):
    scan = api.scan([ipaddress])
    return host.get("port", "n/a")


def phone_info(phonenumber: str) -> dict:
    key =  'UyxAuz0QBfGhzm69yhOhQEhpp7cSZ42j'
    countries = {'US', 'CA'};

    #custom feilds
    additional_params = {
        'country' : countries
    }
    url = 'https://www.ipqualityscore.com/api/json/phone/%s/%s' %(key, phonenumb
er)
    x = requests.get(url, params = additional_params)
    return (json.loads(x.text))

def phone_info(phonenumber: str) -> dict:
    key = 'UyxAuz0QBfGhzm69yhOhQEhpp7cSZ42j'
    countries = {'US', 'CA'}

    # Custom fields
    additional_params = {
        'country': ','.join(countries)  # Convert set to comma-separated string
    }

    # Construct the URL
    base_url = f'https://www.ipqualityscore.com/api/json/phone/{key}/{phonenumbe
r}'
    query_string = urllib.parse.urlencode(additional_params)
    url = f'{base_url}?{query_string}'

    # Make the request
    with urllib.request.urlopen(url) as response:
        # Read and decode the response
        response_data = response.read().decode('utf-8')

    # Parse and return the JSON response
    return json.loads(response_data)

tools = [
    Tool(
        name="ip_quality_score",
        func=phone_info,
        description="useful when you need find information about a phone number.
",
    ),
    Tool(
	name="virus_total",
	func=virus_total,
	description="use to figure out if a url is malware.",
	),
    Tool(
    	name="censys",
    	func=censys_find_location,
    	description="use to find the location of a ip address",
    	),
    Tool(
        name="shodan",
        func=subset_shodan,
        description="useful when you need to figure out information about ip add
ress.",
    ),
    Tool(
        name="wolfram",
        func=wolfram.run,
        description="useful for calculations and mathematical quesions.",
    ),
    Tool(
        name="python_repl",
        func=python_repl,
        description="use this when asked about writing code.",
    ),
    Tool(
        name="ShellTool",
        func=shell_wrapper,
        description="use this to execute shell commands or to find out ip addres
ses from hostnames",
    ),
]


prompt = """The following is a conversation between a human and an AI. The AI is
 talkative and provides information about a target system, organization and doma
in. A user will give information about a hostname or an ip address.  The AI can 
write code and execute it.  If the AI doesn't know the answer to a question, it 
truthfully says it does not know. You have access to the following tools: """


#suffix = (
#    "Begin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {in
p#ut}\n{agent_scratchpad}"
#    ""
#)

message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", ttl=600, session_id="my-session"
)

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history
)

llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
#chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=3, verbose=True)

def _handle_error(error) -> str:
    return str(error)[:50]


# model = ChatOpenAI(temperature=0)
# planner = load_chat_planner(model)
# executor = load_agent_executor(model, tools, verbose=True)
# agent = PlanAndExecute(memory=memory, planner=planner, executor=executor, verb
ose=True)

agent_chain = initialize_agent(
    tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    # max_iterations=30,
    # early_stopping_method="generate",
    memory=memory,
    handle_parsing_errors=True,
    max_tokens=30000, #Giving a maximum of 2768 for queries by the agent. 
)

# local_agent_chain = initialize_agent(
#     tools,
#     llm=local_llm,
#     agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     #agent  = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     #agent = AgentType.OPENAI_FUNCTIONS,
#     verbose=True,
#     #max_iterations=30,
#     #early_stopping_method="generate",
#     memory=memory,
#     handle_parsing_errors=True,
#     max_tokens=4000

# )

# SMS messaging tools and endpoint


def query_agent(query_str: str):
    try:
        response = agent_chain.run(input=query_str)

    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").remove
suffix(
            "`"
        )
    return str(response)