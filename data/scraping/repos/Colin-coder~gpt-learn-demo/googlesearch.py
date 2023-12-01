import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType
from translatechain import translateToChineseTool

OPENAI_API_BASE='https://newcon.space/v1'

if __name__ == "__main__":
 
    # 加载 OpenAI 模型
    llm = OpenAI(temperature=1,max_tokens=2048,api_base=OPENAI_API_BASE,api_key = os.environ.get('OPENAI_API_KEY')) 

    # 加载 serpapi 工具
    # tools = load_tools(["serpapi"], serpapi_api_key='2ac1b6ebd86bcec2def4fbb89bebd57cb4237677db8dd1e24e9a24ecb273c7b3')
    tools = [translateToChineseTool(llm = llm)]

    # 如果搜索完想在计算一下可以这么写
    # tools = load_tools(['serpapi', 'llm-math'], llm=llm)

    # 如果搜索完想再让他再用python的print做点简单的计算，可以这样写
    # tools=load_tools(["serpapi","python_repl"])

    # 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # 运行 agent
    finalAnswer = agent.run("西安今天天气怎么样")
    print(f"nkprint:{finalAnswer}")

