from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

if __name__ == '__main__':
    prompt = "Where are the next summer olympics going to be hosted? What is the population of that country ?"

    # We don't want it to get creative!
    llm = OpenAI(temperature=0)

    tools = load_tools(["ddg-search", "wikipedia"])

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    print(agent.run(prompt))
