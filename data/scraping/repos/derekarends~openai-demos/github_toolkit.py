"""Github Toolkit."""
from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from github_tool import GithubTool
from github_api import GithubApiWrapper


class GithubToolKit(BaseToolkit):
    """Github Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_github_api_wrapper(cls, github_api_wrapper: GithubApiWrapper) -> "GithubToolKit":
        operations = github_api_wrapper.list()
        tools = [
            GithubTool(
                name=operation["name"],
                description=operation["description"],
                mode=operation["mode"],
                api_wrapper=github_api_wrapper,
            )
            for operation in operations
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
