from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    template = """given the full name {person_name} of a person, I want you to get me a link to their LinkedIn profile page.
    Your answer should contain only a URL, not other comments"""

    agent_tools = [
        Tool(
            name="Crawl Google for Linkedin profile page",
            func=get_profile_url,
            description="useful when you need to get the likedin profile of a person",
        )
    ]

    agent = initialize_agent(
        tools=agent_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(template=template, input_variables=["person_name"])

    linkedin_profile_url = agent.run(prompt_template.format_prompt(person_name=name))

    return linkedin_profile_url
