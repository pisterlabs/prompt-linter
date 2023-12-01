from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema.language_model import BaseLanguageModel

from tools.serpapi import search
from tools.regex import get_twitter_username


def get_twitter_profile_username(llm: BaseLanguageModel, name: str) -> str:
    """
    Given a name of a person, returns their Twitter profile username
    :param llm: Base language model to use for COT prompting
    :param name: Name of a person to search
    :return: url
    """
    template = """
    Given a person name {name}, I want you to get me the username from their Twitter profile page URL.
    Your answer should only contain the Twitter @username.
    """

    tools_for_agent = [
        Tool(
            name="Search Google for Twitter profile page",
            func=search,
            description="Use this tool, when you need to find the Twitter profile page URL",
        ),
        Tool(
            name="Extract Twitter username from Twitter profile URL",
            func=get_twitter_username,
            description="Use this tool, when you need to extract Twitter username from Twitter profile URL",
        ),
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(input_variables=["name"], template=template)
    return agent.run(prompt_template.format_prompt(name=name))
