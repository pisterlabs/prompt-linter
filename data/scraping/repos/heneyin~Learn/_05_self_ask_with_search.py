"""
https://python.langchain.com/docs/modules/agents/agent_types/self_ask_with_search

This walkthrough showcases the Self Ask With Search chain.
"""

import env

from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

llm = OpenAI(temperature=0)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
self_ask_with_search.run(
    "What is the hometown of the reigning men's U.S. Open champion?"
)