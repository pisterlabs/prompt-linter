from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType
from dotenv.main import load_dotenv
import os

load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']
serpapi_api_key = os.environ['SERP_API_KEY']

# 加载 OpenAI 模型
llm = OpenAI(temperature=0, max_tokens=2048, openai_api_key=openai_api_key)

 # 加载 serpapi 工具
tools = load_tools(["serpapi"], serpapi_api_key=serpapi_api_key)
# 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 运行 agent
agent.run("京东美股昨天的股价是多少？再帮我评论下")