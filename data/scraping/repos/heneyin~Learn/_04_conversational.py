"""
会话

这里示范如何使用 agent 优化你的对话。
其他 agent 通常被优化为使用工具来找出最佳响应，这在对话环境中并不理想，因为你可能希望 agent 也能与用户交互聊天。

这是通过一种特定类型的 agent （conversational-react-description）来实现的，该 agent 期望与内存组件一起使用。
"""

import env

from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")

llm=OpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

q="hi, i am bob"
result = agent_chain.run(input=q)
print("q: ", q)
print("a:", result)
print("-------------------")

q="what's my name?"
result = agent_chain.run(input=q)
print("q: ", q)
print("a:", result)
print("-------------------")

q="what are some good dinners to make this week, if i like thai food?"
result = agent_chain.run(q)
print("q: ", q)
print("a:", result)
print("-------------------")

q= "tell me the last letter in my name, and also tell me who won the world cup in 1978?"
result = agent_chain.run(input="q")
print("q: ", q)
print("a:", result)
print("-------------------")


q= "whats the current temperature in pomfret?"
result = agent_chain.run(input=q)
print("q: ", q)
print("a:", result)
print("-------------------")


