# The core idea of agents is to use an LLM to choose a sequence of actions to take. In chains, a sequence of actions
# is hardcoded (in code). In agents, a language model is used as a reasoning engine to determine which actions to
# take and in which order.


from config import *
from langchain.llms import OpenAI
from langchain.agents import AgentType, Agent, Tool, initialize_agent, load_tools

llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.3)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

result = agent.run('How old is MS Dhoni?')
print(result)
