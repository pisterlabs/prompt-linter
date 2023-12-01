from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
import requests
import openai
from api_list import get_nft_info,get_nft_metrics
from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
import os

openai.api_key  = os.environ['OPENAI_API_KEY']

# 使用tools自定义工具
nft_metrics_tool = Tool(
    name = "nft_metrics",
    func = get_nft_metrics,
    description = "This function sends a GET request to the NFT information API "
                  "to retrieve information about an NFT with the given contract address and token ID."

)
# 加载llm，用于agent中的语言模型
llm = OpenAI(temperature=0)

# 加载一些工具，便于agent选择使用
tools = load_tools(["llm-math"], llm=llm) + [nft_metrics_tool]


# 初始化agent，给定工具、语言模型和agent的类型
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run()