from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI


llm = ChatOpenAI(temperature=0.5)
math_llm = OpenAI(temperature=0.5)
tools = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)

agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

agent_chain.run("what is my math problem and its solution")
