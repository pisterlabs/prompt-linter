from langchain import OpenAI

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools, initialize_agent
from botchan import settings

DEFAULT_LOAD_TOOLS = [
    # math
    "llm-math",  # mathtool via llm
    # knowledge
    # "wolfram-alpha",
    "wikipedia",
    "arxiv",
    # news
    "news-api",
    # search
    "serpapi",  # search via serp api
]


def create_default_agent():
    llm = OpenAI(temperature=0)
    tools = load_tools(DEFAULT_LOAD_TOOLS, llm=llm, news_api_key=settings.NEWS_API_KEY)

    mrkl = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    return mrkl
