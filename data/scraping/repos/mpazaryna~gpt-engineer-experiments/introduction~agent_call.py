# agent_call.py
from langchain import ConversationChain, PromptTemplate
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents.load_tools import get_all_tool_names
from langchain.llms import OpenAI

# --------------------------------------------------------------
# Agents: Dynamically Call Chains Based on User Input
# --------------------------------------------------------------


def run_agent_demo():
    llm = OpenAI()
    get_all_tool_names()
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    result = agent.run(
        "In what year was python released and who is the original creator? Multiply the year by 3"
    )
    return result
