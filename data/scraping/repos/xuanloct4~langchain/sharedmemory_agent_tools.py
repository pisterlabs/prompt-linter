import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain import LLMChain, PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper
template = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""

prompt = PromptTemplate(
    input_variables=["input", "chat_history"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")
readonlymemory = ReadOnlySharedMemory(memory=memory)
summry_chain = LLMChain(
    llm=llm, 
    prompt=prompt, 
    verbose=True, 
    memory=readonlymemory, # use the read-only memory to prevent the tool from modifying the memory
)

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name = "Summary",
        func=summry_chain.run,
        description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary."
    )
]
prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
agent_chain.run(input="What is ChatGPT?")

agent_chain.run(input="Who developed it?")
agent_chain.run(input="Thanks. Summarize the conversation, for my daughter 5 years old.")
print(agent_chain.memory.buffer)

## This is a bad practice for using the memory.
## Use the ReadOnlySharedMemory class, as shown above.

template = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""

prompt = PromptTemplate(
    input_variables=["input", "chat_history"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")
summry_chain = LLMChain(
    llm=llm, 
    prompt=prompt, 
    verbose=True, 
    memory=memory,  # <--- this is the only change
)

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name = "Summary",
        func=summry_chain.run,
        description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary."
    )
]

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

agent_chain.run(input="What is ChatGPT?")

agent_chain.run(input="Who developed it?")

agent_chain.run(input="Thanks. Summarize the conversation, for my daughter 5 years old.")

print(agent_chain.memory.buffer)