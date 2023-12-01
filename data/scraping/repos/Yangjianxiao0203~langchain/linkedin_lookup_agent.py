from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


def lookup(name: str, description: str = "") -> str:
    """
    agent 流程： 1. 外界的tool是google serach api，找url的
    大模型的reAct：
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    template = """
    given the full name of {name_of_person}, and some description {description_of_person}.
    I want you to get it me a link to their Linkedin profile page. Your answer should be only a url. 
    The urls should be in the format of https://www.linkedin.com/in/<profile-id>
    """
    prompt = PromptTemplate(
        input_variables=["name_of_person", "description_of_person"], template=template
    )

    crawl_tool = Tool(
        name="linkedin_crawler",
        description="This tool crawls linkedin with input name and description and returns a url to the profile page",
        func=get_profile_url,
    )
    tools = [crawl_tool]  # tools 就像action的一环

    agent = initialize_agent(
        tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, llm=llm
    )

    linkedin_url = agent.run(
        prompt.format(name_of_person=name, description_of_person=description)
    )
    return linkedin_url


if __name__ == "__main__":
    template = """
    given the full name of {name_of_person}, and maybe some description {description_of_person}.
    I want you to get it me a link to their Linkedin profile page. Your answer should be only a url. 
    The urls should be in the format of https://www.linkedin.com/in/<profile-id>
    """
    prompt = PromptTemplate(
        input_variables=["name_of_person", "description_of_person"], template=template
    )
    prompt_str = prompt.format(
        name_of_person="Jianxiao Yang",
        description_of_person="He studies in Boston University and has intenship experience in Uber",
    )
    print(prompt_str)
