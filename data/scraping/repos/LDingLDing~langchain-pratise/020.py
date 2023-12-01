from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(temperature=0.0)
tools = load_tools(["arxiv"])

agent_chain = initialize_agent(
  tools, 
  llm, 
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True
)

agent_chain.run("介绍一下2005.14165这篇论文的创新点?")