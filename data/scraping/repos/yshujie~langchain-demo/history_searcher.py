from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

def searchHistoryOfToday():
    # 加载 OpenAI 的模型
    llm = OpenAI(temperature=0, max_tokens=2048)
    
    # 加载 serpapi 工具
    tools = load_tools(['serpapi'])
    
    # 工具加载后需要初始化，verbose 参数为 True，会打印全部的执行过程
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    # 运行 agent
    agent.run(
        "What's the date today? What great events have taken place today in history?"
    )
