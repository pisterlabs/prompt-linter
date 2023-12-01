from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    template = """given the full name {name} I want you to get it me a link to their Linkedin profile page.
                          Your answer should contain only a URL"""

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need get the Linkedin Page URL",
        )
    ]
    agent = initialize_agent(
        tools=tools_for_agent,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        verbose=True,
    )
    prompt_template = PromptTemplate(template=template, input_variables=["name"])
    linkedin_profile = agent.run(prompt_template.format(name=name))

    return linkedin_profile
