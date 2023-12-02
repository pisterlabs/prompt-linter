from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

# Custom packages
from tools.tools import get_profile_url


def lookup(name: str) -> str:
    """Searches for Linkedin or twitter Profile Page by inizializing an agent."""

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    template = """given the full name {name_of_person} I wnat you to get me a link to their Linkedin profile page.
    Your answer should contain only a URL.
    """

    # Here we define the tools that the agent will use.
    # If it the description matches the current task (in some way) the agent will use it.
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need to get the Linkedin page URL",
        )
    ]

    # Agent initialization
    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Prompt template
    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )

    # Run Agent
    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))

    return linkedin_profile_url
