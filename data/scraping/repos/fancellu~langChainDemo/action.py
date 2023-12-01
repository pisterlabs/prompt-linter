from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

if __name__ == '__main__':
    # Action agents

    # LLMs are bad at maths, and would probably get this wrong, even the 3rd president or their DOB can be faulty
    # due to hallucinations
    prompt = "When was the 3rd president of the USA born? What is that year cubed?"

    # We don't want it to get creative!
    llm = OpenAI(temperature=0)

    tools = load_tools(["wikipedia", "llm-math"], llm=llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    print(agent.run(prompt))
