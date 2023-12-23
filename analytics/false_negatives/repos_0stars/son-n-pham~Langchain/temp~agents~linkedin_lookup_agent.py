from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url

import os

openai_api_key = os.environ['OPENAI_API_KEY']
print(openai_api_key)


def lookup(name: str) -> str:

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",
                     openai_api_key=openai_api_key)

    template = """
    Given the full name {name_of_person} I want you to give me a link to their Linkedin profile page.
    Your answer should contain only a URL
    """

    tools_for_agent = [Tool(name="Crawl Google for Linkedin profile page",
                       func=get_profile_url,
                       description='Useful for when you need the Linkedin page URL')
                       ]

    agent = initialize_agent(tools=tools_for_agent,
                             llm=llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)

    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template)

    linkedin_profile_url = agent.run(
        prompt_template.format_prompt(name_of_person=name))

    return linkedin_profile_url
