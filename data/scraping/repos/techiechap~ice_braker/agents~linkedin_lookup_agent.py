 

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI    
from langchain.agents import initialize_agent, Tool, AgentType
from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") 
    template =  """ Given the the full name of person {name_of_person}, I would like to see link of their linkedin profile 
       Your answer should contain only a URL """ 

   
    tools_for_agent = [
        Tool(
             name="Crawl google 4 linkedin profile page", 
             func=get_profile_url,
             description="Useful When you need to get the linkedin profile URL" 
             ),
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
    

    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linkedin_profile_url
