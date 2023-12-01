"""
gpt-3.5-turbo-0613 and gpt-4-0613

在 API 调用中，您可以描述函数并让模型智能地选择输出包含调用这些函数的参数的 JSON 对象。

OpenAI 函数 API 的目标是比通用文本完成或聊天 API 更可靠地返回有效且有用的函数调用

"""

import env

from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)


tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]

# 这里制定使用 openai function
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

result = agent.run("张艺谋导演已经上映的最后一部电影是哪部？哪一年上映的？年份乘以2是多少")
print("result:", result)


