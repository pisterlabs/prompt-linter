"""Twilio Toolkit."""
from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from twilio_tool import TwilioAction
from twilio_api import TwilioApiWrapper


class TwilioToolKit(BaseToolkit):
    """Twilio Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_twilio_api_wrapper(cls, twilio_api_wrapper: TwilioApiWrapper) -> "TwilioToolKit":
        actions = twilio_api_wrapper.list()
        tools = [
            TwilioAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=twilio_api_wrapper,
            )
            for action in actions
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
