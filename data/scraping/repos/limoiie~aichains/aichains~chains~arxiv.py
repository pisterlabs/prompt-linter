from langchain import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

from aichains.annotators import render_output


@render_output
def arxiv(question: str, temperature=0.2, verbose=False):
    """Chat with arxiv."""
    llm = OpenAI(temperature=temperature)
    tools = load_tools(["serpapi", "arxiv"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose
    )

    result = agent.run(question)
    return result
