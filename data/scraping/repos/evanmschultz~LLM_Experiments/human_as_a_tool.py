from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType, AgentExecutor
import os


llm = ChatOpenAI(temperature=0.5)
math_llm = OpenAI(temperature=0.5)
tools: list = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)

agent_chain: AgentExecutor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_chain.run("If I have 5 apples and eat 2, how many do I have left?")
