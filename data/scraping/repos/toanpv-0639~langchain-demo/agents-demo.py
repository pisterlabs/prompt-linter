import os
import requests
from apikey import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain.agents import tool

from langchain.agents import initialize_agent, AgentType

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Text model example

llm = OpenAI(temperature=0.1)

# Define tool for agent


@tool("search", return_direct=True)
def search(query: str) -> str:
    """
    You call this function when user need search information for an input string (in variable query) and return the search results back to user
    """
    return f"Search API Results for {query}"


@tool("post-recommendation", return_direct=True)
def post_recommendation(user_id: str) -> dict:
    """
    You call this function when user want to provide the recommended posts related for user.
    This function will be call to backend API to do this recommendation logic and return most related posts for user
    """
    response = requests.get(
        "https://langchain-demo.free.mockoapp.net/post-recommendation"
    )
    return response.json()


@tool("add", return_direct=True)
def add(a: int, b: int) -> int:
    """
    You call this function when user want to add two number and return the result back to user
    """
    return a + b


@tool("subtract", return_direct=True)
def subtract(a: int, b: int) -> int:
    """
    You call this function when user want to substract two number and return the result back to user
    """
    return a - b


@tool("multiply", return_direct=True)
def multiply(a: int, b: int) -> int:
    """
    You call this function when user want to multiply two number and return the result back to user
    """
    return a * b


tools = [multiply, add, subtract, search, post_recommendation]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

command = "Give me the list of posts related to me"

result = agent.run(command)

print(result)
