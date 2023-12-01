"""
Zero-shot ReAct
https://arxiv.org/pdf/2205.00445.pdf

该代理使用 ReAct 框架仅根据工具的描述来确定要使用哪个工具。可以提供任意数量的工具。该代理要求为每个工具提供描述。

"""

import env

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)


tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# result= agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
result= agent.run("张艺谋导演已经上映的最后一部电影是哪部？哪一年上映的？年份乘以2是多少")
print("result:", result)

"""
Using chat models
"""

# from langchain.chat_models import ChatOpenAI
#
# chat_model = ChatOpenAI(temperature=0)
# agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")

