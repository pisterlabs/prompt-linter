import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

from typing import List, Optional
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.requests import Requests
from langchain.tools import APIOperation, OpenAPISpec
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.agent_toolkits import NLAToolkit

# # Select the LLM to use. Here, we use text-davinci-003
# llm = OpenAI(temperature=0, max_tokens=700) # You can swap between different core LLM's here.

speak_toolkit = NLAToolkit.from_llm_and_url(llm, "https://api.speak.com/openapi.yaml")
klarna_toolkit = NLAToolkit.from_llm_and_url(llm, "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/")

# Slightly tweak the instructions from the default agent
openapi_format_instructions = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: what to instruct the AI Action representative.
Observation: The Agent's response
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
Final Answer: the final answer to the original input question with the right amount of detail

When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response."""


natural_language_tools = speak_toolkit.get_tools() + klarna_toolkit.get_tools()
mrkl = initialize_agent(natural_language_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                        verbose=True, agent_kwargs={"format_instructions":openapi_format_instructions})

mrkl.run("I have an end of year party for my Italian class and have to buy some Italian clothes for it")
# Spoonacular API Console
import os
spoonacular_api_key = os.environ.get("SPOONACULAR_API_KEY") # Copy from the API Console
requests = Requests(headers={"x-api-key": spoonacular_api_key})
spoonacular_toolkit = NLAToolkit.from_llm_and_url(
    llm, 
    "https://spoonacular.com/application/frontend/downloads/spoonacular-openapi-3.json",
    requests=requests,
    max_text_length=1800, # If you want to truncate the response text
)

natural_language_api_tools = (speak_toolkit.get_tools() 
                              + klarna_toolkit.get_tools() 
                              + spoonacular_toolkit.get_tools()[:30]
                             )
print(f"{len(natural_language_api_tools)} tools loaded.")

# Create an agent with the new tools
mrkl = initialize_agent(natural_language_api_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                        verbose=True, agent_kwargs={"format_instructions":openapi_format_instructions})

# Make the query more complex!
user_input = (
    "I'm learning Italian, and my language class is having an end of year party... "
    " Could you help me find an Italian outfit to wear and"
    " an appropriate recipe to prepare so I can present for the class in Italian?"
)

mrkl.run(user_input)
natural_language_api_tools[1].run("Tell the LangChain audience to 'enjoy the meal' in Italian, please!")
