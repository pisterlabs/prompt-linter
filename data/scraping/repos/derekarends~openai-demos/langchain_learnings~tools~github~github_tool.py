"""
This tool allows agents to interact with the Github Api
and operate on a Github instance.

To use this tool, you must first set as environment variables:
    GITHUB_TOKEN
"""

from typing import Optional
from pydantic import Field

from langchain.tools.base import BaseTool
from github_api import GithubApiWrapper


class GithubTool(BaseTool):
    api_wrapper: GithubApiWrapper = Field(default_factory=GithubApiWrapper)
    mode: str
    name = ""
    description = ""

    def _run(self, instructions: Optional[str]) -> str:
        """Use the Github API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)

    async def _arun(self, _: str) -> str:
        """Use the Github API to run an operation."""
        raise NotImplementedError("GithubTool does not support async")
