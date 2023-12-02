import os
from dotenv import load_dotenv
from typing import List
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI


# Load .env variables
load_dotenv()


# LLM Initialization
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(max_retries=3, temperature=0, # type: ignore
                model_name="gpt-3.5-turbo-0613")


def initialize_agent_with_new_openai_functions(tools: List, is_agent_verbose: bool = True, max_iterations: int = 3, return_thought_process: bool = False):
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=is_agent_verbose,
                             max_iterations=max_iterations, return_intermediate_steps=return_thought_process)

    return agent