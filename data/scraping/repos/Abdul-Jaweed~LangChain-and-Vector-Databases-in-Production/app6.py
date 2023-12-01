from langchain.llms import OpenAI

from langchain.agents import AgentType
from langchain.agents import load_agent
from langchain.agents import initialize_agent

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

google_cse_id = os.getenv("GOOGLE_CSE_ID")
google_api_key = os.getenv("GOOGLE_API_KEY")

GOOGLE_CSE_ID=google_cse_id
GOOGLE_API_KEY=google_api_key

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)

search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iteration=6
)

response = agent("What`s the latest news about the Mars rovers?")
print(response['output'])

