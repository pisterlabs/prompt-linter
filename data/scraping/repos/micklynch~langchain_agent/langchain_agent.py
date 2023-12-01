import os, openai
from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

docstore=DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search"
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup"
    )
]
llm = OpenAI(temperature=0, model_name="text-davinci-002")

react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)

question = "Name the current minister for foreign affairs for Ireland and what is their constituency"
react.run(question)
