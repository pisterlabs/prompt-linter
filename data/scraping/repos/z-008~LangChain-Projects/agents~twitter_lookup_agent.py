from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from tool.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    template = """Given a person fullname {name_of_person}, I want you to find a link to their twitter profile page and extract from there the Twitter username.
                 Your final answer should only contain Twitter username"""
    tools_for_agent = [
        Tool(
            name="Crawl Google for Twitter Profile page",
            func=get_profile_url,
            description="useful when you need to get Twitter Page URL",
        )
    ]
    agent = initialize_agent(
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools_for_agent,
        verbose=True,
    )
    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )
    twitter_username = agent.run(prompt_template.format_prompt(name_of_person=name))
    return twitter_username
