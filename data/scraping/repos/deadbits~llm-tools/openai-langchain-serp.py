#!/usr/bin/env python3
import os

from langchain import LLMChain

from langchain.llms import OpenAI

from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.agents import ZeroShotAgent
from langchain.agents import AgentExecutor

from langchain.memory import ConversationBufferMemory

from langchain.utilities import GoogleSerperAPIWrapper


os.environ['SERPER_API_KEY'] = 'KEY'
os.environ['OPENAI_API_KEY'] = 'sk-KEY'

search = GoogleSerperAPIWrapper()
openai_llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

prefix = 'You are an AI assistant having a conversation with a human. Do your best to answer questions. You have access to the following tools:'
suffix = """Begin!

{chat_history}
Question: {input}
{agent_scratchpad}"""

tools = [
    Tool(
        name='Search',
        func=search.run,
        description='Useful for when you need to answer questions about current events and topics not covered in your knowledge base.'
    )
]

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=['input', 'chat_history', 'agent_scratchpad']
)

memory = ConversationBufferMemory(memory_key='chat_history')


def run_chain(user_prompt):
    # build and run chain
    llm_chain = LLMChain(llm=openai_llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    print(agent_chain.run(user_prompt))


if __name__ == '__main__':
    while True:
        user_input = input('prompt> ')
        if user_input.lower() == '!quit' or user_input.lower() == '!exit':
            break
        run_chain(user_input)

    print('[exiting...]')
