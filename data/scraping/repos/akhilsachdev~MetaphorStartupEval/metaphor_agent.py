
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMMathChain, LLMChain, APIChain
from langchain.prompts import StringPromptTemplate, ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.utilities import MetaphorSearchAPIWrapper
from langchain.agents import Agent, Tool, initialize_agent, AgentType, AgentExecutor, tool, AgentOutputParser
from metaphor_python import Metaphor
from dotenv import load_dotenv
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from typing import List, Dict, Any, Optional, Union
from langchain.agents import OpenAIFunctionsAgent
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

import re
import os

load_dotenv('.env')

openai_api_key = os.getenv('OPENAI_API_KEY')
metaphor_api_key = os.getenv('METAPHOR_API_KEY')

metaphor_client = Metaphor(metaphor_api_key)

@tool
def search(query: str):
    """Call search engine with a query."""
    return metaphor_client.search(query, use_autoprompt=True, num_results=3)

@tool
def get_contents(ids: List[str]):
    """Get contents of a webpage.
    
    The ids passed in should be a list of ids as fetched from `search`.
    """
    return metaphor_client.get_contents(ids)

@tool
def find_similar(url: str):
    """Get search results similar to a given URL.
    
    The url passed in should be a URL returned from `search`
    """
    return metaphor_client.find_similar(url, num_results=3)


metaphor_search = MetaphorSearchAPIWrapper(metaphor_api_key=metaphor_api_key)
llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=openai_api_key, max_tokens=400, temperature=0)
# llm = OpenAI(model='gpt-3.5-turbo', openai_api_key=openai_api_key, max_tokens=500, temperature=0, n=1)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    search, 
    get_contents, 
    find_similar,
    # Tool(
    #     name='calculator',
    #     func=llm_math_chain.run,
    #     description='Calculate information from search results'
    # )
]


prompt = ChatPromptTemplate.from_messages([
    ("system", 
     '''You are a venture capitalist conducting due dilligence on a startup. Find information about the startup and answer the following questions:
        1. Analyze the startup's business model
        2. Compare the startup's competitors
        3. Analyze the startup's market
        4. Evaluate the startup's potential for success

        Use 3 sentences for each question.
     '''),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)

agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps'])
} | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_output = agent_executor.invoke({"input": "Evaluate Ramp's potential"})
# agent_output['output']

# print(agent_output['output'])



