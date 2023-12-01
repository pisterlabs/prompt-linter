import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType


#set env variable for openai and serpapi
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "0d8984b84cfc9a044677dba9b03b595a52f526b2c8e1cb3007fc7d2653a2942d"


#1 Load LLM agent ---

LLM1= OpenAI(temperature=0)

#2 Loading tools ---

tools = load_tools(["serpapi", "llm-math"], llm=LLM1)

#3 Instantiate agent

agent = initialize_agent(tools,LLM1, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("Who is Nishant Shandilya")




