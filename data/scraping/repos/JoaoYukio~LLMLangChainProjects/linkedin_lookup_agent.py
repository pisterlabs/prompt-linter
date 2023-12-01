# from ferramentas.ferramentas import get_profile_url

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.serpapi import SerpAPIWrapper


def get_profile_url(text: str) -> str:
    """Searches for Linkedin profile page"""
    search = SerpAPIWrapper()
    res = search.run(f"{text}")
    return res


def lookup(name: str) -> str:
    """
    Search for a LinkedIn profile and return the link to the profile.
    """

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    template = """
        Dado o nome completo {name_of_person} eu gostaria que encontrasse o link do LinkedIn dessa pessoa.
        Sua resposta só deve ser a URL do LinkedIn.
    """

    tools_for_agent = [
        Tool(
            name="Craw Google 4 linkedin profile page",
            func=get_profile_url,
            description="Util quando você precisa encontrar o link do LinkedIn de uma pessoa.",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Tecnica de raciocinio
        verbose=True,  # Printa o que o agente está fazendo
    )

    promt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    linkedin_profile_url = agent.run(promt_template.format_prompt(name_of_person=name))
    return linkedin_profile_url
