
import os
import pandas as pd


import pandas as pd


from langchain.chat_models import ChatAnthropic
# from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate
from langchain import LLMChain
import cohere
import streamlit as st

os.environ['username'] = 'itsmethefounder@outlook.com'
os.environ['password'] = "Tech4Life!"
os.environ['ANTHROPIC_API_KEY'] = "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"
os.environ['LANGCHAIN_API_KEY'] = "ls__0aa97ffdedf342068430ab83273564fd"





LANGCHAIN_API_KEY = "ls__0aa97ffdedf342068430ab83273564fd"
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = "Foodsmith"


cohere_api_key = "5GIQYhLSWrnXOprlPqJSwKu6l7awxtBfi26R9c7c"


def main_function(user_connections_list, user_query):
 
    claude = ChatAnthropic(temperature=0)
   
    
    user_connections_list_string = user_connections_list.to_string()

    if user_query==None:
        user_query = "Give me people who worked in the software development industry"

    template = """

    Context: You are a personnel recommending agent. Given the user required personnel, find the top 4 people with relevant experience
    and background that fit the user's needs in a table format with also a score on the right on how much they match.

    You will do this analysis from the user's list available below where the list contains all the user's bio's
    You do not need more information, and make judgements solely from the list info below while using the first and last names from the User List below
    User List: {user_connections_list}

  

    User Query : {user_query}

    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["user_connections_list", "user_query"],
        template=template
    )

    claude_chain = LLMChain(prompt=prompt_template, llm=claude)

    claude_output = claude_chain.run(
        user_connections_list=user_connections_list_string, user_query=user_query)

    return claude_output