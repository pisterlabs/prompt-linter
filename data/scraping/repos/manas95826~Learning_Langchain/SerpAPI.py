# https://serpapi.com/
import os
os.environ['SERPAPI_API_KEY'] = "248676f3e794ef77d2c88a239493c2a99a7000eb4b8d051aa4d21daa1125efae"
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
# llm = OpenAI()

# Next, let's load some tools to use.
tools = load_tools(["serpapi"])

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
# We are passing in some example questions and answers to train the agent on the definition of LangChain.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    train_examples=[
        ("What is LangChain ?", "LangChain is a framework designed to simplify the creation of applications using large language models."),
    ],
    verbose=True,
)

# Now let's test it out!
agent.run("What is LangChain ?")
