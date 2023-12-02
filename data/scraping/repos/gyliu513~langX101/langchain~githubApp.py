import os

from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits.github.toolkit import GitHubToolkit
from langchain.llms import OpenAI
from langchain.utilities.github import GitHubAPIWrapper

llm = OpenAI(temperature=0)
github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github)
agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)