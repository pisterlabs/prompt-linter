import ipaddress, os, re, socket
from transformers import AutoTokenizer, AutoModelForCausalLM
from shodan import Shodan
# import pwnlib
from langchain.chat_models import ChatOpenAI
#import cve_searchsploit as CS
from langchain import Wikipedia, OpenAI 
from langchain.llms import PromptLayerOpenAI
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.agents import Tool, AgentType, tool
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, LLMChain
#from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import load_tools, initialize_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents import Tool, AgentExecutor
from langchain.tools.python.tool import PythonREPLTool
from langchain.tools.ifttt import IFTTTWebhook
from langchain.utilities import (
    WikipediaAPIWrapper,
    PythonREPL,
    BashProcess,
    TextRequestsWrapper,
)
from langchain.llms import HuggingFaceHub

memory = ConversationBufferMemory()
wolfram = WolframAlphaAPIWrapper()
wikipedia = WikipediaAPIWrapper()
python_repl = PythonREPLTool()
bash = BashProcess()
requests = TextRequestsWrapper()

def hostname(hostname: str) -> str:
    """useful when you need to get the ipaddress associated with a hostname"""
    try:
       ip = ipaddress.ip_address(addr)
       return ip
    except ValueError:
       return 'Invalid ip address'
    
    

def subset_shodan(addr: str):


    #ipv4_extract_pattern = "(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    #extracted_ip = re.findall(ipv4_extract_pattern, addr)[0]
    shodan_api = Shodan(os.environ.get('SHODAN_API_KEY'))
    
    #if ipaddress.ip_address(extracted_ip).is_private:
    #    return "This is a private ip address."
    addr = addr.replace("scan ", "")
    try:
        host = shodan_api.host(addr)
    except Exception as e:
      return "Shodan has no info"
    ports = ""
    for i in host['data']:
        ports += "Port {} \n".format(i['port'])

    return """
    IP: {}
    Organization: {}
    Operating System: {}
    Country: {}
    Location: Lat {} Long {}
    Asn: {}
    Transport: {}
    Port: {}
    """.format(host['ip_str'],
               host.get('org', 'n/a'),
               host.get('os', 'n/a'),
               host.get('country_name', 'n/a'),
               host.get('lat', 'n/a'),
               host.get('long', 'n/a'),
               host.get('asn', 'n/a'),
               host.get('transport', 'n/a'),
               ports,)

def scan_ip_addr(ipaddress):
    scan = api.scan([ipaddress])
    return host.get("port", "n/a")

def phone_info(phone_number: str) -> str:
    import http.client

    conn = http.client.HTTPSConnection("api.trestleiq.com")
    
    conn.request("GET", "/3.0/phone_intel?api_key=SOME_STRING_VALUE&phone={}&phone.country_hint=US".format(phone_number))

    res = conn.getresponse()
    data = res.read()

    phone_intel_result_payload = data.decode("utf-8")

    
    return result_payload

tools = [
    Tool(
        name="trestle",
        func=hostname,
        description="useful when you need lookup a hostname given an ip address.",
    ),
    Tool(
       name="shodan",
        func=subset_shodan,
        description="useful when you need to figure out information about ip address.",
    ),
    Tool(
        name="wolfram",
        func=wolfram.run,
        description="useful for calculations and mathematical quesions.",
    ),
    Tool(
        name="python_repl",
        func=python_repl.run,
        description="use this when asked about writing code.",
    ),
    Tool(
        name="Bash",
        func=bash.run,
        description="use this to execute shell commands or to find out ip addresses from hostnames",
    ),
]


prefix = """The following is a conversation between a human and an AI. The AI is talkative and provides information about a target system, organization and domain. A user will give information about a hostname or an ip address.  The AI can write code and execute it.  If the AI doesn't know the answer to a question, it truthfully says it does not know. You have access to the following tools: """


suffix = (
    "Begin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {input}\n{agent_scratchpad}"
    ""
)

message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", ttl=600, session_id="my-session"
)

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history
)

llm = ChatOpenAI(temperature=0, model="gpt-4")

def _handle_error(error) -> str:
    return str(error)[:50]
    
#model    = ChatOpenAI(temperature=0)
#planner  = load_chat_planner(model)
#executor = load_agent_executor(model, tools, verbose=True)
#agent = PlanAndExecute(memory=memory, planner=planner, executor=executor, verbose=True)

agent_chain = initialize_agent(
    tools,
    llm=llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #agent  = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    #agent = AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    #max_iterations=30,
    #early_stopping_method="generate",
    memory=memory,
    handle_parsing_errors=True,
    max_tokens=4000

)
llama_feature = False
if llama_feature:
    repo_id = "meta-llama/Llama-2-13b-chat-hf"
    llama2_llm =  HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"max_length": 4000, "temperature": 0.0}
    )

    llama2_agent_chain = initialize_agent(
    tools,
    llm=llama2_llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #agent  = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    #agent = AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    #max_iterations=30,
    #early_stopping_method="generate",
    memory=memory,
    handle_parsing_errors=True,
    max_tokens=4000

)

# SMS messaging tools and endpoint


def query_agent(query_str: str):
    try:
        response = agent_chain.run(input=query_str)
        
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    return str(response)

