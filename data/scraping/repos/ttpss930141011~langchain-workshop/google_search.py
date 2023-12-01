from dotenv import dotenv_values
import os
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType

config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = config.get("SERPAPI_API_KEY")
print(os.environ["OPENAI_API_KEY"])
print(os.environ["SERPAPI_API_KEY"])

# 加載 OpenAI 模型
llm = OpenAI(temperature=0, max_tokens=2048)

# 加載 serpapi 工具
tools = load_tools(["serpapi"])

# 如果搜索完想再計算一下可以這麼寫
# tools = load_tools(['serpapi', 'llm-math'], llm=llm)

# 如果搜索完想再讓他再用python的print做點簡單的計算，可以這樣寫
# tools=load_tools(["serpapi","python_repl"])

# 工具加載後都需要初始化，verbose 參數為 True，會打印全部的執行詳情
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 運行 agent
agent.run(
    "What's the date today? What great events have taken place today in history?")