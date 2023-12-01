from dotenv import load_dotenv
load_dotenv()

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

tools = load_tools(["llm-math"], llm=llm)


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What is 2349 + 234 * 2394 / 123?")

