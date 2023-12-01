import os
import langchain

from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = ""
os.environ['SERPAPI_API_KEY'] = ""

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

search = SerpAPIWrapper()

# Define a list of tools offered by the agent
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful when you need to answer questions about current events. You should ask targeted questions.",
    ),
]

mrkl = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True
)

langchain.debug = True

result = mrkl.run("What is the weather in LA and SF?")
print(result)

# Configuring max iteration behavior
mrkl = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    max_iterations=2,
    early_stopping_method="generate",
)
result = mrkl.run("What is the weather in NYC today, yesterday, and the day before?")
print(result)
