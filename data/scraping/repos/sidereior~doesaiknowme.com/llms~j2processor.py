from langchain.llms import AI21
from langchain.agents import load_tools
from langchain.agents import initialize_agent

llm = AI21(temperature=0.9) 
tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run("tell me about Dan Olexio, from Ohio. Be sure to search multiple social media sources and multiple sources for their online presence.")