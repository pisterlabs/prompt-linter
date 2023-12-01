from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
tools = load_tools(['llm-math', 'human'], llm=llm)

agent = initialize_agent(tools=tools, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         llm=llm,
                         verbose= True)

agent.run("What is 9 times 6")

# working here, just wasnt working in notebook