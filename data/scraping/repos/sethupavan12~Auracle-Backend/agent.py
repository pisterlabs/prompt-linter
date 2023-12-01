from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain
import anthropic
from langchain.llms import Anthropic
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.agents.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatAnthropic


# We'll make a temporary directory to avoid clutter
working_directory = TemporaryDirectory()



search = SerpAPIWrapper()

llm = Anthropic(streaming=True,callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
# llm = ChatAnthropic()
# llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="Useful to learn about best pratice to do something or to know about current events you can use this to search"
    )
    ,
    # Tool(
    # name="human",
    # description="Useful for when you need to get input from a human",
    # func=input,  # Use the default input function
    # )
]

def plan_execute(tools):
    model = ChatAnthropic()
    planner = load_chat_planner(model)


    executor = load_agent_executor(model, tools, verbose=True)

    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True,handle_parsing_errors="Check your output and make sure it conforms!")

    return agent

def plan_execute_with_internet():
    model = ChatAnthropic()
    planner = load_chat_planner(model)
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="Useful to learn about best pratice to do something or to know about current events you can use this to search"
        )
        ,
        # Tool(
        # name="human",
        # description="Useful for when you need to get input from a human",
        # func=input,  # Use the default input function
        # )
    ]

    executor = load_agent_executor(model, tools, verbose=True)
    
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True,handle_parsing_errors="Check your output and make sure it conforms!")

    return agent



