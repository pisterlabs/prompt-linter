from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from toolbox.search_tool import get_profile_url


def lookup(name: str) -> str:
    # Step-1: Engineer the prompt. Must provide output indicator: just the URL
    prompt = """ Given the full name {name_of_person} I want you to find me a link to thier twitter profile page and extract from it their username. In your final answer you return only the person's username.
    """

    # Step-2: Get instance of the LLM.
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    """  Step-3:
     1. Create the toolbox (List of tools) for the agent.
     2. The tool contains:
        - The name of the tool, MUST be UNIQUE between every tools
        - The functionality/behavior of the tool:
            -- This function will be called if the agent decides to use this tool.
        - Descripton of the tool
            -- When agent searches what tool to use, it uses description of the tool
    """
    tool_box_for_the_agent = [
        Tool(
            name="Crawl google for twitter profile page",
            func=get_profile_url,
            description="This tool is useful when you need to get the twitter page url.",
        )
    ]

    """ Step-4: Check the Agent section of the langchain documentation
    1. Create the Agent.
        - Toolbox
        - LLM
        - Agent Type
            -- Determines the process in which the reasoning will be done
            
     2. verbose = True: Logs the reasoning process
    """
    agent = initialize_agent(
        tools=tool_box_for_the_agent,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
    )

    """ Step-5: 
    1. Create a prompt template
    """
    prompt_template = PromptTemplate(
        template=prompt, input_variables=["name_of_person"]
    )

    """ Step-6: 
    1. Run the agent
    """
    twitter_username = agent.run(prompt_template.format_prompt(name_of_person=name))

    return twitter_username
