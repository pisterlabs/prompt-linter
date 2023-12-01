#
# Plan and execute agent
#
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
# from langchain import SerpAPIWrapper
# from langchain.agents.tools import Tool added from langchain.agents
from langchain import LLMMathChain
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from subprocess import Popen, PIPE
from dotenv import load_dotenv
load_dotenv()

llm = OpenAI( temperature=0 ) # 1st things 1st, we need an llm
### DEFINE PARTS FOR TOOLS // these need to be defined before we build the tools array ###
class ShellSchema(BaseModel):
    command: str = Field( description="The shell command to execute on a wsl 2 ubuntu linux subsystem for windows 10 followed by a redirect of stderr to dev null." )
    # 2>&1 otherwise we pay for tokens to read a bunch of "Permission Denied" errors!

def shell_function(command: str) -> str:
    process = Popen(command, stdout=PIPE, shell=True)
    output, _ = process.communicate()
    return output.decode()

# search = SerpAPIWrapper() # not sure which search to use
search = GoogleSearchAPIWrapper()
llm_math_chain = LLMMathChain.from_llm( llm=llm, verbose=True)
###

### BUILD THE TOOLS ARRAY  // Now we can build the tools array ###
tools = [
    Tool(
        name="shell",
        func=shell_function,
        args_schema=ShellSchema,
        description="useful for when you need to execute shell commands"
    ),
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math" )]
###


### BUILD THE CHAIN  // Now we can build the chain ###

### continue setting up CppMaker from https://python.langchain.com/docs/modules/memory/agent_with_memory
prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"  

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt( tools, prefix=prefix, suffix=suffix, input_variables=[ "input", "chat_history", "agent_scratchpad" ])
memory = ConversationBufferMemory( memory_key="chat_history" )
llm_chain = LLMChain( llm=OpenAI( temperature=0 ), prompt=prompt )
agent = ZeroShotAgent( llm_chain=llm_chain, tools=tools, verbose=True )
agent_chain = AgentExecutor.from_agent_and_tools( agent=agent, tools=tools, verbose=True, memory=memory )

### end creating chain.  uncomment some of the following to test it out.

### END BUILDING THE CHAIN ###


# Define the ExceptionHandlerAgent class
# class ExceptionHandlerAgent:
#     def __init__(self, executor, verbose=False):
#         self.executor = executor
#         self.verbose = verbose

#     def run(self, plan: str):
#         # For simplicity, directly executing the plan using the executor
#         # In a more advanced scenario, this might involve parsing the plan,
#         # determining the appropriate tool to use, and then executing it
#         return self.executor(plan)

# Define the tool schema for the ExceptionHandler
# class ExceptionHandlerSchema(BaseModel):
#     exception: Exception = Field(description="The exception instance to handle.")
#     context: str = Field(description="The context or scenario where the exception occurred.")

# class ExceptionHandlerSchema(BaseModel):
#     exception: str
#     context: str

# Define the tool function for the ExceptionHandler
# def handle_exception(exception: Exception, context: str) -> str:
#     print(f"Exception caught in context '{context}': {exception}")
#     return f"An error occurred while {context}. Please check the inputs and try again."
# llm = OpenAI( temperature=0 )
# model    = ChatOpenAI( temperature=0 )
# planner  = load_chat_planner(model)
# executor = load_agent_executor(model, tools, verbose=True)

# Initialize the PlanAndExecute agent
# agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# Tools for the ExceptionHandlerAgent
# exception_tools = [
#     Tool(
#         name="Exception Handler",
#         func=handle_exception,
#         args_schema=ExceptionHandlerSchema,
#         description="Handles exceptions and provides user-friendly messages." )]

# Initialize the ExceptionHandlerAgent
# exception_agent_executor = load_agent_executor(model, exception_tools, verbose=True)
# exception_agent = ExceptionHandlerAgent(executor=exception_agent_executor, verbose=True)

# Example of running the PlanAndExecute agent with integrated exception handling
# try:
#     agent.run( """- Search the web for the latest ai automation news.\n- Pick the latest news.\n- Summarize that piece of news and put it in a professional-looking HTML document using gray and white alternating colors.\n-include in the html document images from the original article or source.\n- Give the document a name relevant to the content and store it in the current directory.""" )
### this ran successfully one time, but no html output.

prompt = """Please help me fix the following error: ``` error Thank you.  Please help me fix the following error: Exception has occurred: ImportError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
attempted relative import with no known parent package
  File "/home/adamsl/.pyenv/versions/3.10.6/lib/python3.10/site-packages/pydantic/main.py", line 29, in <module>
    from .class_validators import ValidatorGroup, extract_root_validators, extract_validators, inherit_validators
  File "/home/adamsl/.pyenv/versions/3.10.6/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/adamsl/.pyenv/versions/3.10.6/lib/python3.10/runpy.py", line 196, in _run_module_as_main (Current frame)
    return _run_code(code, main_globals, None,
ImportError: attempted relative import with no known parent package ``` """

try:
    agent_chain.run( input=prompt )
    # agent.run( """- Read this html link: ``` link http://127.0.0.1:5500/pexpect/doc/index.template.html ``` and output an html summary of the contents of the link.""" )

except Exception as e:
    print ( f"- Handle the exception '{str(e)}' that occurred while executing the plan." )
