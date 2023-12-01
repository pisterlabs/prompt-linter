import os
from dotenv import load_dotenv
import openai
from typing import List
from langchain.agents import initialize_agent,AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from models import llm_model_type


load_dotenv()

openai.api_key= os.getenv('OPENAI_API_KEY')
# google_serp_api.client = os.getenv('SERPAPI_API_KEY')
# google_api_key = os.getenv('GOOGLE_CSE_ID')
# google_cse_id = os.getenv('CUSTUM_SEARCH_ID')

lili = ChatOpenAI(max_retries=3, temperature=0,model_name =llm_model_type)


def initialize_agent_zero_shot(tools:List, is_agent_verbose: bool =True,max_iterations:int = 3, return_thought_process: bool = False ):
    memory = ConversationBufferMemory(memory_key="chat_history")

    agent = initialize_agent(
        tools,lili,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=is_agent_verbose,
        max_iterations=max_iterations,
        return_intermediate_steps=return_thought_process,
        memory=memory
    )
    return agent

