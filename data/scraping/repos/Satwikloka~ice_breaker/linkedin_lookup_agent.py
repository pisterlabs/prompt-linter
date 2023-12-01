from tools.tools import get_profile_url

from langchain import PromptTemplate
from langchain.chat_models import vertexai

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType


def lookup(name: str) -> str:
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": 0.8,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 40,
    }

    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                          Your answer should contain only a URL"""
    tools_for_agent1 = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need get the Linkedin Page URL",
        ),
    ]

    agent = initialize_agent(
        tools_for_agent1, chat_model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )
    linkedin_username = agent.run(prompt_template.format_prompt(name_of_person=name))

    return linkedin_username
