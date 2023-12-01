# fmt: off
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain import PromptTemplate
from tools import get_profile_url

# fmt: on

# NOTE: The agent uses information from the doc string to perform the task.
# important to note that the tool is what allows the agent to interact with the
# outside world. The agent is the brain, the tool is the body.

# function input string name, return linkedin profile url


def profile_lookup(name: str) -> str:
    """Lookup a LinkedIn profile by name."""
    # create an instance of chatopenai model
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # build template
    template = """ given the person full name {person_name}, I want you to go to the persons linkedin profile 
    and return only the url of the profile."""

    # define agent tool
    tools_for_agent = [
        Tool(
            name="crawl Google for LinkedIn profile",
            func=get_profile_url,
            description="useful for when you need to get the LinkedIn profile of a person",
        )
    ]

    # initialize agent
    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        Verbose=True,  # should provide reasoning process of the agent
    )

    # define agent prompt template
    prompt_template = PromptTemplate(
        template=template, input_variables=["person_name"])

    # linkedin url
    url = agent.run(prompt_template.format_prompt(person_name=name))
    return url
