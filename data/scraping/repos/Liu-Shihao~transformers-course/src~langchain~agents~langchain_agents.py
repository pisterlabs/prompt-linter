from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
"""
自定义agent中所使用的工具

自定义工具里面有个比较有意思的地方，使用哪个工具的权重是靠 工具中描述内容 来实现的，和我们之前编程靠数值来控制权重完全不同。

比如 Calculator 在描述里面写到，如果你问关于数学的问题就用他这个工具。我们就可以在上面的执行过程中看到，他在我们请求的 prompt 中数学的部分，就选用了Calculator 这个工具进行计算。
"""
llm = OpenAI(temperature=0)

# 初始化搜索链和计算链
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# 创建一个功能列表，指明这个 agent 里面都有哪些可用工具，agent 执行过程可以看必知概念里的 Agent 那张图
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]

# 初始化 agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 执行 agent
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
