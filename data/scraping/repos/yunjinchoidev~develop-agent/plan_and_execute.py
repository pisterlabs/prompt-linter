from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain
from load_dotenv import load_dotenv
from langchain.tools import PythonREPLTool

load_dotenv()

search = SerpAPIWrapper()
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    # Tool(
    #     name="Calculator",
    #     func=llm_math_chain.run,
    #     description="useful for when you need to answer questions about math",
    # ),
    Tool(
        name="python_repl",
        func=PythonREPLTool(),
        description="python_repl, useful for when you need to run python code, and get the output",
    ),
    Tool(
        name="python_repl",
        func=PythonREPLTool(),
        description="save the output to a file",
    ),
]

model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
agent.run("tetris python code")
