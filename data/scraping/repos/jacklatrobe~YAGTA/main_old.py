# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## main.py - main program loop for YAGTA

# Base Imports
import os
import sys
import logging
from collections import deque
from typing import Dict, List, Optional, Any

# Logging - Initialise
logging.basicConfig(encoding='utf-8', level=logging.INFO)

# Langchain Imports
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate, OpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun

from langchain_experimental.autonomous_agents import BabyAGI


# Vectorstore - Imports
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

# Vectorstore - Configuration
embeddings_model = OpenAIEmbeddings().embed_query
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

# OpenAI LLM - The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
if "OPENAI_API_KEY" not in os.environ:
    logging.critical("Env OPENAI_API_KEY not set - exiting")
    sys.exit(1)


# BabyAGI - Program main loop
def main():
    # OpenAI LLM - Initialise
    llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-16k", max_tokens=2000)

    OBJECTIVE = "What events are on in Melbourne in September 2023?"


    # BabyAGI - Define tool functions
    writing_prompt = PromptTemplate.from_template(
            "You are a writer given the following task and context: {objective}\n"
            "Produce a high quality piece of writing or text that achieves this objective, making sure to keep any included context or information in your response."
        )
    writing_chain = LLMChain(llm=llm, prompt=writing_prompt, verbose=True)
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    duckduckgo = DuckDuckGoSearchRun()
    todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at breaking down objectives into step by step tasks."
    "You must write all lists of tasks, in priority order, in this format:"
    "1. Task 1"
    "2. Task 2"
    "3. Task 3"
    "Respond with nothing but a list of tasks that you could follow to do this objective: {objective}"
    )
    todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)

    # BabyAGI - Define agent tools
    tools = [
        Tool(
            name="Wikipedia Search",
            func=wikipedia.run,
            description="useful for searching wikipedia for facts about events, companies or concepts. Input: a search query. Output: search results.",
        ),
        Tool(
            name="DuckDuckGo Search",
            func=duckduckgo.run,
            description="useful for searching the internet for information using duckduckgo. Input: a search query. Output: search results",
        ),
        Tool(
            name="Writing Tool",
            func=writing_chain.run,
            description="useful for writing other texts. Input: an objective to write about. Output: a piece of writing",
        ),
        Tool(
            name="Planner",
            func=todo_chain.run,
            description="useful for when you need to come up with plan how to achieve an objective. Input: an objective to create a todo list for. Output: a numbered todo list for that objective. Please be very clear what the objective is!",
        ),
    ]

    # BabyAGI - Define manager prompts
    prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )

    # BabyAGI - Set up the execution agent
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    # OpenAI LLM - Logging of LLMChains
    verbose = False

    # BabyAGI - Max Iterations, If None, will keep on going forever
    max_iterations: Optional[int] = None

    # BabyAGI - Initialise
    baby_agi = BabyAGI.from_llm(
        llm=llm, 
        vectorstore=vectorstore, 
        task_execution_chain=agent_executor,
        verbose=verbose, 
        max_iterations=max_iterations,
        handle_parsing_errors="Check your output and formatting!",
    )

    # BabyAGI - Run
    baby_agi({"objective": OBJECTIVE})

    ## NOTE: KNOWN ISSUE with LangChain's implementation of BabyAGI currently storing duplicate result_ids in vectorstore leading to error and halt:
    ## https://github.com/langchain-ai/langchain/issues/7445
    ## Not worth fixing, fix already submitted waiting for merge to main
    

if __name__ == "__main__":
    main()
