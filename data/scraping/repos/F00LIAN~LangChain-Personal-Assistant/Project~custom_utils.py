import logging
import os
from dotenv import load_dotenv
from twilio.rest import Client
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMMathChain
from langchain.agents.agent_toolkits import O365Toolkit
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from typing import Optional, Type
from langchain.agents.agent_toolkits import ZapierToolkit
import json
from langchain.tools import YouTubeSearchTool
from langchain.utilities.zapier import ZapierNLAWrapper
from youtube_search import YoutubeSearch
from langchain.utilities import SerpAPIWrapper
from langchain.tools import YouTubeSearchTool
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.utilities import TextRequestsWrapper
from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


# Set environment variables
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)
twilio_number = os.getenv("TWILIO_NUMBER")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
zapier_nla_api_key = os.getenv("ZAPIER_NLA_API_KEY")

# Set Environ API keys
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
youtube = YouTubeSearchTool()
zapier = ZapierNLAWrapper()


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Conversational Memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)


# Sending message logic through Twilio Messaging API
def send_message(to_number, body_text):
    try:
        message = client.messages.create(
            from_=f"whatsapp:{twilio_number}",
            body=body_text,
            to=f"whatsapp:{to_number}",
        )
        logger.info(f"Message sent to {to_number}: {message.body}")
    except Exception as e:
        logger.error(f"Error sending message to {to_number}: {e}")


# Creating Custom Tools for the Agent


class CustomSearchTool(BaseTool):
    name = "custom_search"
    description = "useful for when you need to answer questions about current events"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return search.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("search does not support async")


class YoutubeTool(BaseTool):
    name = "Youtube Videos"
    description = (
        "Use this tool when you need to lookup videos and send links for reference."
    )

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return youtube.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("search does not support async")


class ZapierTool(BaseTool):
    name = "Zapier Tools"
    description = "use this tool when you need to create and add data to a google spreadsheet. Also, use this tool when you need to make a post on twitter."

    def _run(
        self,
        query: str,
        input: str,
        type: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        llm = OpenAI(temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))
        toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        tools = toolkit.get_tools()
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )

        return agent.run(query)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class Office365Tool(BaseTool):
    name = "Office365Emails"
    description = "use this tool when you need to send, write, and draft emails. Also, use this tool when you need to create events on a calender"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        toolkit = O365Toolkit()
        tools = toolkit.get_tools()
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )

        return agent.run(query)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


# Initialize the agent
def agent_creation(query):
    tools = [CustomSearchTool(), Office365Tool(), ZapierTool()]
    llm = OpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        # memory=conversational_memory,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
    )

    return agent.run(query)
