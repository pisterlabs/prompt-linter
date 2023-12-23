from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool,AgentType,initialize_agent
from tools.customtools import get_profile_id

# Tool will suply the agents with execution functionalities



def lookup_linkedin(name):
    llm = ChatOpenAI(temperature=0,model_name = 'gpt-3.5-turbo')

    template = "Given the full name {name_of_person} I want you to get me the profile url from their linked_in page. Your answer should only contain the profile url"
    
    tools_for_agent =[Tool(name = "Crawl Google 4 linkedin profile page",func=get_profile_id, description="Useful for when you need to get the linkedin profile url")]
    
    agent = initialize_agent(tools=tools_for_agent,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True) 

    prompt_template = PromptTemplate(template=template,input_variables=['name_of_person'])

    profile_id =agent.run(prompt_template.format_prompt(name_of_person =name))  
    
    return f"{profile_id}" 
