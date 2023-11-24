from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # I found that the answers could be wrong if the template below has any spaces or line breaks. Had to put
    # everything in a single line
    template = """given the fullname  {name_of_person} I want you to get me a link to their Linkedin profile page. 
    Your answer should contain only a URL"""

    # The first parameter is name, which is mandatory. We gave it Crawl Google for LinkedIn profile page. Now this
    # must be unique between every tool. So the function is also mandatory and it's the function that's going to be
    # called when the LM decides to use this tool. We're going to put in the future a function which is going to
    # search Google for that LinkedIn page. The description argument is optional, but it's super recommended to use
    # it. Because the agent, when it determines whether to use this tool or not, it does it through the description.
    # if we don't fill it out, it would be much harder for the agent to decide whether the tool is worth using or
    # not. And the behavior we will get will be unexpected and not what we want. So it's very important to be very
    # explicit about this and to write a clear explanation that describes the tool as it is.

    tools_for_agent = [
        Tool(
            name="crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need to get the Linkedin URL",
        )
    ]

    # Agent type is a super important parameter because it will decide what is the reasoning process that the agent
    # is going to make to perform a task The agent will read the description above Verbose=true will make the agent
    # to be aware before each task it does and prints out the reasoning process using REACT

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    linked_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linked_profile_url
