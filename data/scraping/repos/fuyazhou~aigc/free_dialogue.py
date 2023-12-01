import os
import langchain
from config import *
from util import *
from langchain.llms import OpenAI, Cohere, HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from typing import Optional, Type
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent, tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from logging import getLogger

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

prompt = PromptTemplate(
    input_variables=["text"],
    template="{text}",
)

llm = OpenAI(temperature=0)
chat = ChatOpenAI(temperature=0)

llm_chain = LLMChain(llm=llm, prompt=prompt)
chat_model_chain = LLMChain(llm=chat, prompt=prompt)

logger = getLogger()


class CustomSearchTool(BaseTool):
    name = "search tool"
    description = "一个搜索引擎。 当你需要回答有关实时的问题或者计算的问题调用该工具，否则不要使用该工具。 输入应该是搜索查询。"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return search(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


# You can create the tool to pass to an agent
chat_tool = Tool(
    name="Chat",
    description="一个非常有用的助理，你可以回答除了实时问题或者计算问题以外的任何问题，用中文回答问题",
    func=chat_model_chain.run,
    return_direct=True

)


def get_free_dialogue_answer(user_id, query):
    try:
        logger.info(f"******** get_free_dialogue_answer ***************")
        logger.info(f"user_id = {user_id}, user_query = {query} ")
        tools = [chat_tool, CustomSearchTool()]
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        result = agent.run(query)
        logger.info("******** get_free_dialogue_answer  done ***************")
        logger.info(f"user_id = {user_id}, user_query = {query} , response= {result}")
        return result
    except Exception as e:
        logger.warning(f"An error occurred during dialogue processing:{e}")
        return common_responses


if __name__ == '__main__':
    query = "北京时间"
    user_id = "122324"
    res = get_free_dialogue_answer(user_id, query)
    print(str(res))
