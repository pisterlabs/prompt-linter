import os
import sys
from threading import Lock
from typing import Any, List, Optional

from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory
from langchain.utilities import SerpAPIWrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import get_settings
from langchain.pydantic_v1 import BaseModel, Field
from langchain_model.api2d_model import Api2dLLM
from vectordbs.faissdb import FaissDb
from vectordbs.vectordb import VectorDb


class CustomerQuestion(BaseModel):
    #matching with the key in LLMChain
    query: str = Field()

class SalesChain:
    #For thread safe singleton example see [here](https://refactoring.guru/design-patterns/singleton/python/example#example-1)
    _instance = None
    _lock: Lock = Lock()

    _tools: List[Tool]
    _agent: AgentExecutor

    def __new__(cls,*args, **kwargs):
        # The overriden __new__ need to have *args, **kwargs to pass param to __init__
        with cls._lock:
            if cls._instance is None:
                """
                Here super() is calling the 'object'class whose constructor cannot take more args > 1.
                And there's no point in calling object.__new__() with more than a class param thus it throws an exception.
                See [stackoverflow](https://stackoverflow.com/questions/59217884/new-method-giving-error-object-new-takes-exactly-one-argument-the-typ)
                """
                cls._instance = super().__new__(cls) #
        return cls._instance

    def __init__(self, tools: Optional[List[Tool]] = None, memory: Optional[BaseMemory] = None):
        vectordb = FaissDb()
        llm = Api2dLLM(temperature=0)
        if tools is not None:
            self._tools = tools
        else:
            self._tools = self._default_tools(vectordb, llm)
        #TODO Fix exception when using vectordb's memory
        memory = memory if memory is not None else vectordb.createMemory()
        self._agent = self._create_agent(memory, self._tools, llm)

    def _default_tools(self, vectordb: VectorDb, llm: LLM) -> List[Tool]:
        web_tool = Tool.from_function(
            #TODO Improve web search by switching to google shop searching with more input params
            func=SerpAPIWrapper(params = {
                "engine": "google",
                "location": "Austin, Texas, United States",
                "google_domain": "google.com",
                "gl": "cn",
                "hl": "zh-cn",
                "tbm": "shop"
            }).run,
            name="Web Search",
            description="""useful for when you could not find proper answer from 'VectorDb QA Search' tool \n
            and need to answer questions about product specifications and market price."""
            # coroutine= ... <- you can specify an async method if desired as well
        )
        vectorqa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.8, "k": 1}
            )
        )
        vectorqa_tool = Tool.from_function(
            func=vectorqa_chain.run,
            name="VectorDb QA Search",
            description=" useful for searching existing electronic device sales questions and answers. you should always use this first.", #Emphasize on priority
            #args_schema=CustomerQuestion
            # coroutine= ... <- you can specify an async method if desired as well
        )
        return [vectorqa_tool, web_tool]

    def _create_agent(self, memory: BaseMemory, tools: List[Tool], llm: LLM) -> AgentExecutor:
        #prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools: """
        prefix = """你是一个专业而有礼貌的的电器销售人工智能体，优先使用"VectorDb QA Search"工具(注意不更改input的问题)，尽可能回答问题："""
        suffix = """开始!"

        {chat_history}
        客户问题: {input}
        {agent_scratchpad}"""
        
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain)
        return AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory,
            # `I now know the final answer and can provide it to the customer.` is causing LLM output parse issue.
            handle_parsing_errors="Print out LLM output, try parsing it and make sure it conforms!" #This helps to reduce eranous CoT
        )
    
    @property
    def agent(self):
        return self._agent
    
if __name__ == "__main__":
    #TODO Fix "Observation: Invalid or incomplete response" causing infinit looping on ReAct
    """Reference: 
    [Different language in instructions](https://github.com/langchain-ai/langchain/issues/8867)
    [Missing Action after Thought](https://github.com/langchain-ai/langchain/issues/12689)
    [Handle parsing errors in langchain](https://python.langchain.com/docs/modules/agents/how_to/handle_parsing_errors)
    [Wierd chain of thoughts](https://github.com/langchain-ai/langchain/issues/2840)
    [Observation: Invalid or incomplete response](https://github.com/langchain-ai/langchain/issues/9381)
    """
    #TODO Fix failure to call VectoDb QA Search with exact input unchanged.(Prevent GPT from summarizing the use's question)
    text = SalesChain(memory=ConversationBufferMemory(memory_key="chat_history")).agent.run(input = "A100 GPU卡的价格多少？")
    print(text)