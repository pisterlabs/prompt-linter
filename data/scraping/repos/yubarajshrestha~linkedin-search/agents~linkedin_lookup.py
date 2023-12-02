from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    template = """
        given the full name {name} I want you to get me a link to their LinkedIn profile page.
        your answer should contain only a URL.
    """

    tools_for_agent = [Tool(
        name="Crawl google 4 linkedin profile page",
        func=get_profile_url,
        description="Useful for when you need to get the LinkedIn profile page URL",
    )]

    agent = initialize_agent(tools=tools_for_agent, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    prompt = PromptTemplate(
        input_variables=["name"],
        template=template,
    )

    linkedin_profile_url = agent.run(prompt.format_prompt(name=name))

    return linkedin_profile_url
