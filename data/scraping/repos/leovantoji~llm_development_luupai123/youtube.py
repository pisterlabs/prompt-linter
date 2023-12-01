from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.tools import YouTubeSearchTool


tool = YouTubeSearchTool()

tools = [
    Tool(
        name="Search",
        func=tool.run,
        description=(
            """useful for when you need to give links to youtube videos. """
            """Remember to put https://youtube.com/ in front of """
            """every link to complete it"""
        ),
    )
]


agent = initialize_agent(
    tools,
    OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

agent.run("Whats a joe rogan video on an interesting topic")
