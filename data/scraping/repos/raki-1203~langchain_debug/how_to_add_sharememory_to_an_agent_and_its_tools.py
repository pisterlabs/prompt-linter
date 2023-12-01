import os

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY
os.environ['SERPER_API_KEY'] = c.SERPER_API_KEY
os.environ['GOOGLE_API_KEY'] = c.GOOGLE_API_KEY
os.environ['GOOGLE_CSE_ID'] = c.GOOGLE_CSE_ID


if __name__ == '__main__':
    template = """This is a conversation between a human and a bot:
    
{chat_history}

Write a summary of the conversation for {input}:
"""

    prompt = PromptTemplate(
        input_variables=['input', 'chat_history'],
        template=template,
    )

    memory = ConversationBufferMemory(memory_key='chat_history')
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    summary_chain = LLMChain(
        llm=OpenAI(),
        prompt=prompt,
        verbose=True,
        memory=readonlymemory,  # use the read-only memory to prevent the tool from modifying the memory
    )

    search = GoogleSearchAPIWrapper()
    tools = [
        Tool(
            name='Search',
            func=search.run,
            description='useful for when you need to answer questions about current events',
        ),
        Tool(
            name='Summary',
            func=summary_chain.run,
            description='useful for when you summarize a conversation. The input to this tool should be a string,'
                        'representing who will read this summary.',
        ),
    ]

    prefix = """Have a conversation with a human, answering the following questions as best you can. You
have access to the following tools:"""
    suffix = """Begin!

{chat_history}
Question: {input}
{agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=['input', 'chat_history', 'agent_scratchpad'],
    )

    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

    print(agent_chain.run(input="What is ChatGPT?"))

    print(agent_chain.run(input="Who developed it?"))

    print(agent_chain.run(input='Thanks. Summarize the conversation, for my daughter 5 years old.'))

    print(agent_chain.memory.buffer)

