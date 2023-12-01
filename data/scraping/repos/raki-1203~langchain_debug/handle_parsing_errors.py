import os

from langchain import OpenAI, LLMMathChain, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents.types import AGENT_TO_CLASS

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY


def _handle_error(error) -> str:
    return str(error)[:50]


if __name__ == '__main__':
    # Setup
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name='Search',
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions.",
        ),
    ]

    # Error
    # mrkl = initialize_agent(
    #     tools,
    #     ChatOpenAI(temperature=0),
    #     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    # )
    #
    # mrkl.run("Who is Leo DiCaprio's girlfriend? No need to add Action")

    # Default error handling
    # mrkl = initialize_agent(
    #     tools,
    #     ChatOpenAI(temperature=0),
    #     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     handle_parsing_errors=True,
    # )
    #
    # mrkl.run("Who is Leo DiCaprio's girlfriend? No need to add Action")

    # Custom Error Message
    # mrkl = initialize_agent(
    #     tools,
    #     ChatOpenAI(temperature=0),
    #     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     handle_parsing_errors="Check your output and make sure it conforms!",
    # )
    #
    # mrkl.run("Who is Leo DiCaprio's girlfriend? No need to add Action")

    # Custom Error Function
    mrkl = initialize_agent(
        tools,
        ChatOpenAI(temperature=0),
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=_handle_error,
    )

    mrkl.run("Who is Leo DiCaprio's girlfriend? No need to add Action")












