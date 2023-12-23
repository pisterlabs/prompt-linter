from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    template = """
        Given the full name {name_of_person} I want you to get me a link to their Linkedin profile page.
        you answer should contain only a URL"""

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Linkedin profile page",
            func=get_profile_url,
            description="Useful for when you need to get the Linkedin Page URL",
        )
    ]

    agent = initialize_agent(
        llm=llm,
        tools=tools_for_agent,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linkedin_profile_url
