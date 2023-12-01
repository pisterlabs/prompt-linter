"""Jira Toolkit."""
from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from jira_tool import JiraAction
from jira_api import JiraApiWrapper


class JiraToolkit(BaseToolkit):
    """Jira Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_jira_api_wrapper(cls, jira_api_wrapper: JiraApiWrapper) -> "JiraToolkit":
        actions = jira_api_wrapper.list()
        tools = [
            JiraAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=jira_api_wrapper,
            )
            for action in actions
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
