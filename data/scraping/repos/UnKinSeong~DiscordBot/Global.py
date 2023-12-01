from __future__ import annotations
import os
from dotenv import load_dotenv
from cllm_tools import *
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import traceback
import dotenv

dotenv.load_dotenv('.env')


# Token File Position
TOKEN_LOCATION = "token_count.csv"

# LLM Configuration
if os.environ.get('openai_api_type') == 'azure':
    llm = AzureChatOpenAI(deployment_name=os.environ.get('openai_deployment_name'),
                      model_name=os.environ.get('openai_model_name'),
                      openai_api_base=os.environ.get('openai_api_base'),
                      openai_api_version=os.environ.get('openai_api_version'),
                      openai_api_key=os.environ.get('openai_api_key'),
                      openai_api_type=os.environ.get('openai_api_type'),
                      temperature=0.0
                      )
else:
    llm = ChatOpenAI(
        model=os.environ.get('openai_model_name'),
        openai_api_key=os.environ.get('openai_api_key'),
        temperature=0.0,
    )

tools = [
    utc_clock,
    file_reader,
    file_writer,
    mkdir,
    use_glob,
]
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=9,
    return_messages=True
)
agentType = AgentType.OPENAI_FUNCTIONS
verbose = True
max_iterations = 7
early_stopping_method = 'generate'

# Agent Construction
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=agentType,
    max_iterations=max_iterations,
    verbose=verbose,
    early_stopping_method=early_stopping_method,
    memory=memory,
)
