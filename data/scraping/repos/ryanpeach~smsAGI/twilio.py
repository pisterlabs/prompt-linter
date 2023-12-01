# Download the helper library from https://www.twilio.com/docs/python/install
import asyncio
import os

from langchain.agents import tool
from langchain.schema import AIMessage
from langchain.tools import BaseTool
from sqlalchemy.orm import Session
from twilio.rest import Client

from lib.sql import SuperAgent, ThreadItem

# Set environment variables for your credentials
# Read more at http://twil.io/secure
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

WAITING_FOR_USER_RESPONSE = "Waiting for user response..."


# TODO: Hope to merge these into one tool eventually, but don't want to
#       multi argument parser right now.
class SendMessageTool(BaseTool):
    name = "send_message"
    description = """
    Sends a message to the user. Does not wait for a response.
    """

    def __init__(self, session: Session, super_agent: SuperAgent):
        super()
        self.session = session
        self.super_agent = super_agent

    def _run(self, query: str) -> str:
        """Use the tool."""
        message = str(query)
        client.messages.create(
            body=message,
            from_=os.environ["TWILIO_FROM_PHONE_NUMBER"],
            to=os.environ["TWILIO_TO_PHONE_NUMBER"],
        )
        ThreadItem.create(
            session=self.session,
            super_agent=self.super_agent,
            msg=AIMessage(content=message),
        )
        return "You have sent the message to the user. You may or may not receive a response. You may ask other questions without waiting for a response. You may also send other messages to the user without waiting for a response. "

    async def _arun(self, query: str) -> str:
        return self._run(query=query)


class SendMessageWaitTool(BaseTool):
    name = "send_message_wait"
    description = """
    Sends a message to the user. Waits for a response.
    """

    def __init__(self, session: Session, super_agent: SuperAgent):
        super()
        self.session = session
        self.super_agent = super_agent

    def _run(self, query: str) -> str:
        message = str(query)
        client.messages.create(
            body=message,
            from_=os.environ["TWILIO_FROM_PHONE_NUMBER"],
            to=os.environ["TWILIO_TO_PHONE_NUMBER"],
        )
        ThreadItem.create(
            session=self.session,
            super_agent=self.super_agent,
            msg=AIMessage(content=message),
        )
        self.super_agent.wait_for_response = True
        return WAITING_FOR_USER_RESPONSE

    async def _arun(self, query: str) -> str:
        return self._run(query=query)
