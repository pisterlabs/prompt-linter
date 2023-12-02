from langchain.llms import VertexAI
from tools.tools import get_google_search, scrape_linkedin_profile
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import GmailToolkit


gmail_toolkit = GmailToolkit()
gmail_tools = [gmail_toolkit.get_tools()[0]]


def get_gmail_agent() -> AgentExecutor:
    llm = VertexAI(temperature=0.5, verbose=True, max_output_tokens=1000)

    agent = initialize_agent(
        gmail_tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent


def get_search_agent() -> AgentExecutor:
    llm = VertexAI(temperature=0, verbose=True, max_output_tokens=1000)

    tools_for_agent = [
        Tool(
            name="GoogleSearch",
            func=get_google_search,
            description="useful for when you need get a google search result",
        ),
        Tool(
            name="scrape_linkedin_profile",
            func=scrape_linkedin_profile,
            description="useful for getting information on a Linkedin profile url",
        ),
    ]

    agent = initialize_agent(
        tools_for_agent,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    return agent
