from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
   
def searchWeather():
    # first load the language model
    llm = OpenAI(temperature=0)
    
    # then load the tools
    tools = load_tools(['serpapi', 'llm-math'], llm=llm)
    
    # then initialize the agents
    agent = initialize_agent(llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    agent.run("What was the high temperature is SF yesterday in Fahrenheit? What is that number raised to the .023 power?")

    