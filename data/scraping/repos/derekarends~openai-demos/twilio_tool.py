"""
This tool allows agents to interact with the Twilio Api
and operate on a Twilio instance.

To use this tool, you must first set as environment variables:
    SLACK_BOT_TOKEN
"""

from typing import Optional
from pydantic import Field

from langchain.tools.base import BaseTool
from twilio_api import TwilioApiWrapper


class TwilioAction(BaseTool):
    api_wrapper: TwilioApiWrapper = Field(default_factory=TwilioApiWrapper)
    mode: str
    name = ""
    description = ""

    def _run(self, instructions: Optional[str]) -> str:
        """Use the Twilio API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)

    async def _arun(self, _: str) -> str:
        """Use the Twilio API to run an operation."""
        raise NotImplementedError("TwilioAction does not support async")
