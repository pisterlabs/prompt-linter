import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


from dotenv import load_dotenv

load_dotenv()

def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name=os.environ.get('MODEL_NAME'))
    template = """given the name {name_of_person} I want you to find a link to their Twitter profile page, and extract from it their username.
       In Your Final answer only the person's username"""

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Twitter profile page",
            func=get_profile_url,
            description="useful for when you need get the Twitter Page URL",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    twitter_username = agent.run(prompt_template.format_prompt(name_of_person=name))

    return twitter_username
