"""
This tool allows agents to interact with the Slack Api
and operate on a Slack instance.

To use this tool, you must first set as environment variables:
    SLACK_BOT_TOKEN
"""

from typing import Optional
from pydantic import Field

from langchain.tools.base import BaseTool
from slack_api import SlackApiWrapper


class SlackTool(BaseTool):
    api_wrapper: SlackApiWrapper = Field(default_factory=SlackApiWrapper)
    mode: str
    name = ""
    description = ""

    def _run(self, instructions: Optional[str]) -> str:
        """Use the Slack API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)

    async def _arun(self, _: str) -> str:
        """Use the Slack API to run an operation."""
        raise NotImplementedError("SlackTool does not support async")
