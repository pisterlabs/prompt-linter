from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    template = """given the full name {name_of_person}, I want you to get me a link to their 
    LinkedIn profile page. Your answer should contain only the URL, do not include anything like I have found... Make sure your response is only the url - this is very important."""

    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )

    tools_for_agent = [
        Tool(
            name="Crawl google for linkedin profile page.",
            func=get_profile_url,
            description="Useful for getting linkedin page url",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    # print(linkedin_profile_url)

    return linkedin_profile_url
