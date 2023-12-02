name = "📊 CSV Agent"
arguments = ["csvs"]
annotated = ["ZeroShot Agent","Default LLM","REPL Tool"]



from langchain.agents import create_csv_agent
from langchain.agents import AgentType
from models.llms.llms import *

def agent(csvs):
    agent_executor = create_csv_agent(llm,csvs,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    
    return agent_executor