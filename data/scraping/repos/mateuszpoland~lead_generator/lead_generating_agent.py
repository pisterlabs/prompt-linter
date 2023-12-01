import os
from dotenv import load_dotenv

from langchain import PromptTemplate, LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool, Tool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")