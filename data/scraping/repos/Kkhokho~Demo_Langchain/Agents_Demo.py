import os
import requests
from langchain.llms import OpenAI
from langchain.agents import tool

from langchain.agents import initialize_agent, AgentType

os.environ["OPENAI_API_KEY"] = "API_KEY"

llm = OpenAI(temperature=0.3)

# Define tool for agent

@tool("search", return_direct=True)
def search(query: str) -> str:
    """
    You call this function when user need search information for an input string (in variable query) and return the search results back to user
    """
    return f"Search API Results for {query}"


tools = [search]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

command = "Give me the list of posts related to me"

result = agent.run(command)

print(result)