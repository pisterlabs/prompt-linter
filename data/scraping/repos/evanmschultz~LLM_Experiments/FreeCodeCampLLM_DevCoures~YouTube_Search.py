from langchain.tools import YouTubeSearchTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType, AgentExecutor
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
import os


tool = YouTubeSearchTool()

tools: list[Tool] = [
    Tool(
        name="Search",
        func=tool.run,
        description="Useful for when you need to give links to youtube videos. Remember to put https://youtube.com/ in front of every link to complete it",
    )
]


agent: AgentExecutor = initialize_agent(
    tools=tools,
    llm=OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent.run("How do you avoid over dependency in python class construction?")
