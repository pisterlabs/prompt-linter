"""A Plan and Execute agent using the langchain framework

based on the following example: https://python.langchain.com/docs/modules/agents/agent_types/plan_and_execute
"""
import os

# Langchain imports
from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain

# Custom tool imports
from tools.overpass.tool import OverpassQueryTool



serpapi_api_key = os.getenv("SERPAPI_API_KEY")

search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
overpass_sequential_chain = OverpassQueryTool.

tools = [
    Tool(
        name="OverpassQueryTool",
        func=overpass_sequential_chain.run,
        description="useful for when you need to look for things and places and their properties",
    ),
    
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]

model = ChatOpenAI(temperature=0)

planner = load_chat_planner(model)

executor = load_agent_executor(model, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run(
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
)
