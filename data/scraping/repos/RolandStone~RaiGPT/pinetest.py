import datetime
import json
import openai
import os
import pandas as pd
import pinecone
import re
from tqdm.auto import tqdm
from typing import List, Union
import zipfile

# Langchain imports
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
# LLM wrapper
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
# Conversational memory
from langchain.memory import ConversationBufferWindowMemory
# Embeddings and vectorstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# Vectorstore Index
index_name = 'podcasts'
api_key = "4b54cc12-aa94-4362-bb99-304dd4f181e1"

# find environment next to your API key in the Pinecone console
env = "asia-southeast1-gcp-free"

pinecone.init(api_key=api_key, environment=env)
pinecone.whoami()
pinecone.list_indexes()
