import os
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


os.environ["OPENAI_API_KEY"] = ""


# search = GoogleSearchAPIWrapper()
template = """{input}"""

prompt = PromptTemplate(input_variables=["input"], template=template)

llm_chain = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=prompt,
    verbose=True,
    # memory=ConversationBufferWindowMemory(k=2),
)

tools = [
    Tool(
        name="LLM",
        func=llm_chain.run,
        description="useful for when you need to answer questions about current events",
    ),
    # Tool(
    #     name="Canada",
    #     func=llm_chain.run,
    #     description="useful for when you need to answer questions about Canada",
    # ),
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
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt, verbose=True)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

agent_chain.run(input="How many people live in canada?")

agent_chain.run(input="what is their national anthem called?")
