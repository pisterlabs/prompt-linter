from langchain.llms import OpenAI
from langchain.agents import load_tools,  ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv(".env.local")

llm = OpenAI(temperature=0, verbose=True)
tools = load_tools(["wikipedia", "llm-math"],llm=llm)

# creating prompt for agent
prefix = """AI agent has following tools for external supports"""
suffix = """Context Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
    # input param will come from user input
    # chat_history param will come from memory
    # from langchain doc: The prompt in the LLMChain must include a variable called "agent_scratchpad" where the agent can put its intermediary work.
)

memory = ConversationBufferMemory(memory_key="chat_history") # this key must match with prmopt

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools,memory=memory)




input_text = ""
while True:
    input_text = input("[You]: ")
    if input_text.lower() == "quit":
        break
    output_text = agent_chain.run(input_text)
    print("[Bot]:", output_text)




# Sample I/O
# [You]: Where is Bangladesh?
# [Bot]: Bangladesh is located in South Asia, bordered by India to the west, north, and east, and Myanmar to the southeast; to the south it has a coastline along the Bay of Bengal.
# [You]: What is the capital of it?
# [Bot]: The capital of Bangladesh is Dhaka.
# [You]: quit