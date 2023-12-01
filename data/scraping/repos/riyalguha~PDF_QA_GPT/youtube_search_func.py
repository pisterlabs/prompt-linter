from langchain.tools import YouTubeSearchTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
import os

os.environ['OPENAI_API_KEY'] = "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"

tool = YouTubeSearchTool()

tools = [
    Tool(
        name="Search",
        func=tool.run,
        description="useful for when you need to give links to youtube videos. Remember to put https://youtube.com/ in front of every link to complete it",
    )
]

agent = initialize_agent(
    tools,
    OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
)

agent.run("what is joe rogan video on an interesting topic")