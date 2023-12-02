import os
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
# from langchain.experimental.plan_and_execute import (
#     PlanAndExecute,
#     load_agent_executor,
#     load_chat_planner,
# )
from langchain.prompts import MessagesPlaceholder

from langchain.chains import LLMChain
from langchain.llms import AzureOpenAI, OpenAI
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

import logging
from datetime import datetime
from api.chain_tools import load_tools



class ArcGISTutor:
    def __init__(self, llm_type="openai", session_id="my-session"):
        logging.info("loading ArcGISTutor")
        self.__set_llm__(llm_type)
        self.redis_url = os.getenv("REDIS_URL")
        self.session_id = session_id



    def __set_llm__(self, llm_type="openai"):
        if llm_type == "openai":
            self.llm = OpenAI(temperature=0)
            self.model = ChatOpenAI(temperature=0)
        elif llm_type == "azure":
            self.llm = AzureOpenAI(temperature=0)
            self.model = AzureChatOpenAI(temperature=0)
        else:
            raise ValueError("llm_type must be 'openai' or 'azure'")

    def get_prompt(self, tools):
        tool_names = ", ".join([tool.name for tool in tools])
        
        prefix = """Have a conversation with a human, answering the following questions as best you can. Here are some tips:
        1. No need to wait for user's response.
        2. No need to repeat a single tool more than twice.
        3. Must Return with the "Final Answer:" once you are done with the conversation.
        4. You have access to the following tools:"""
        suffix = """Begin!"

        {memory}
        Question: {input}
        {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "memory", "agent_scratchpad"],
        )
        return prompt

    def agent(self):
        message_history = RedisChatMessageHistory(
            url=self.redis_url, ttl=600, session_id=self.session_id
        )
        memory = ConversationBufferWindowMemory(k=20, return_messages=True, 
                                                chat_memory=message_history, 
                                                memory_key="memory")
        tools = load_tools(self.llm, memory)
        prompt = self.get_prompt(tools)
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }
        llm_chain = LLMChain(llm = self.model, prompt=prompt, verbose=True)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, memory=memory, agent_kwargs=agent_kwargs
        )
        # agent_chain = PlanAndExecute(
        #     planner=planner,
        #     executor=executor,
        #     verbose=True,
        #     prompt=self.prompts(),
        #     memory=memory,
        # )
        return agent_chain


tutor = ArcGISTutor()
agent = tutor.agent()

print("ArcGISTutor loaded")
# agent.run(
#     input="Hi I am Bob"
    
# )
# agent.run(
#     input="What is my name?",
# )
