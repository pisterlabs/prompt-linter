#
# Plan and execute agent
#
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
# from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain
from langchain.tools import Tool
from pydantic import BaseModel, Field
from subprocess import Popen, PIPE
from dotenv import load_dotenv
load_dotenv()

# Define the ExceptionHandlerAgent class
class ExceptionHandlerAgent:
    def __init__(self, executor, verbose=False):
        self.executor = executor
        self.verbose = verbose

    def run(self, plan: str):
        # For simplicity, directly executing the plan using the executor
        # In a more advanced scenario, this might involve parsing the plan,
        # determining the appropriate tool to use, and then executing it
        return self.executor(plan)

# Define the tool schema for the ExceptionHandler
# class ExceptionHandlerSchema(BaseModel):
#     exception: Exception = Field(description="The exception instance to handle.")
#     context: str = Field(description="The context or scenario where the exception occurred.")

class ExceptionHandlerSchema(BaseModel):
    exception: str
    context: str

# Define the tool function for the ExceptionHandler
def handle_exception(exception: Exception, context: str) -> str:
    print(f"Exception caught in context '{context}': {exception}")
    return f"An error occurred while {context}. Please check the inputs and try again."

class ShellSchema(BaseModel):
    command: str = Field( description="The shell command to execute on a wsl 2 ubuntu linux subsystem for windows 10 followed by a redirect of stderr to dev null." )
    # 2>&1 otherwise we pay for tokens to read a bunch of "Permission Denied" errors!

def shell_function(command: str) -> str:
    process = Popen(command, stdout=PIPE, shell=True)
    output, _ = process.communicate()
    return output.decode()

# search = SerpAPIWrapper()
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="shell",
        func=shell_function,
        args_schema=ShellSchema,
        description="useful for when you need to execute shell commands"
    ),
    # Tool(
    #     name = "Search",
    #     func=search.run,
    #     description="useful for when you need to answer questions about current events"
    # ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math" )]

model    = ChatOpenAI( temperature=0 )
planner  = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)

# Initialize the PlanAndExecute agent
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# Tools for the ExceptionHandlerAgent
exception_tools = [
    Tool(
        name="Exception Handler",
        func=handle_exception,
        args_schema=ExceptionHandlerSchema,
        description="Handles exceptions and provides user-friendly messages." )]

# Initialize the ExceptionHandlerAgent
exception_agent_executor = load_agent_executor(model, exception_tools, verbose=True)
exception_agent = ExceptionHandlerAgent(executor=exception_agent_executor, verbose=True)

# Example of running the PlanAndExecute agent with integrated exception handling
# try:
#     agent.run( """- Search the web for the latest ai automation news.\n- Pick the latest news.\n- Summarize that piece of news and put it in a professional-looking HTML document using gray and white alternating colors.\n-include in the html document images from the original article or source.\n- Give the document a name relevant to the content and store it in the current directory.""" )
### this ran successfully one time, but no html output.

try:
    agent.run( """- Read this html link: ``` link http://127.0.0.1:5500/pexpect/doc/index.template.html ``` and output an html summary of the contents of the link.""" )

except Exception as e:
    print ( f"- Handle the exception '{str(e)}' that occurred while executing the plan." )