
import os
import environment
import langchain
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from llms import defaultLLM as llm

## Agent run with tracing. Ensure that OPENAI_API_KEY is set appropriately to run this example.
# print(os.environ["OPENAI_API_KEY"])

tools = load_tools(["llm-math"], llm=llm())


agent = initialize_agent(
    tools, llm(), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("What is 2 raised to .123243 power?")

# Agent run with tracing using a chat model
agent = initialize_agent(
    tools, ChatOpenAI(temperature=0), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("What is 2 raised to .123243 power?")
