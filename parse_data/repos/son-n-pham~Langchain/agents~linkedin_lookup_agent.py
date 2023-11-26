from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

import os
openai_api_key = os.environ['OPENAI_API_KEY']


def lookup(name: str):
    pass
