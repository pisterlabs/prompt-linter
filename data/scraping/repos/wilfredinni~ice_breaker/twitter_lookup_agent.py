from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    """Lookup a person on LinkedIn and return their profile data."""
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """
    Given the full name {full_name} I want you to get me the link to their
    Twitter profile page and extract the username. Your answer should contain
    only the username
    """

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Twitter profile page",
            func=get_profile_url,
            description="Useful when you need to get a Tweeter profile URL",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["full_name"],
    )

    twitter_username = agent.run(
        prompt_template.format_prompt(full_name=name),
    )
    return twitter_username
