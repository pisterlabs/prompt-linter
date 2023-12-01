from keys import mykey, serpkey

import os

from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = mykey
os.environ["SERPAPI_API_KEY"] = serpkey

from langchain.utilities import SerpAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.agents import AgentType, initialize_agent, load_tools, Tool

llm = OpenAI(temperature=0.4)

def chooseandsearch(s):
    # s = f"What are the best {cuisine} resturants in {loc}, just tell me the names?"
    print(s)

    params = {
        "engine": "bing",
        "gl": "us",
        "location": "Milpitas, California, United States",
        "hl": "en",
    }
    search = SerpAPIWrapper(params=params)

    tools = [
        Tool(
        name="SERPAPI Answer",
        description="chains from the search and location hopefully",
        func=search.run,
    )
    ]
    agent  = initialize_agent(tools,llm, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
    l = (agent.run(s))
    return l

# p = chooseandsearch("Indian", "Milpitas, CA")

