import os
from langchain.agents import Tool, AgentExecutor
from langchain import SerpAPIWrapper, LLMChain
from typing import List
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain

# Define which tools the agent can use to answer user queries
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]

def create_agent(name, custom_prefix, custom_suffix):
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=custom_suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    name = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )
    return name

prefix = """You are a member of an incidence response team, all your knowledge is centered around incidence response. You have access to the following tools:

"""

# Define custom prefixes and suffixes for each agent
ben_suffix = """Your name is Ben you are the (CEO)
Use the following format:
Question: the input question you must address what it affects in the company
Thought: you should always think about what it affects in the company
Action Input: the input to the action, 
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: talk about the effects to the company and ask a question to provide more clarity on the problem

{chat_history}
Question: {input}
{agent_scratchpad}"""

tyne_suffix = """Your name is Tyne you are the  (CTO) 
As the Chief Technology Officer (CTO) of a forward-thinking tech company, you play a pivotal role in shaping the company's technological vision and driving innovation. Your responsibilities span from overseeing technology development and research to ensuring that your company remains at the forefront of emerging trends.

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: suggest the final answer to the original input question and explain deeply and also ask a peering question to seek more understanding

{chat_history}
Question: {input}
{agent_scratchpad}"""

da_suffix = """Your name is Da you are the (Assistant)
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: suggest the final answer to the original input question

{chat_history}
Question: {input}
{agent_scratchpad}"""

# Create agents with custom templates
ben_agent = create_agent("Ben", prefix, ben_suffix)
tyne_agent = create_agent("Tyne", prefix, tyne_suffix)
da_agent = create_agent("DA", prefix, da_suffix)

def main():
    # Simulate a conversation for incident response
    print("Incident Response Simulation:")
    while True:
        user_input = input("You (Incident Manager): Enter your message: ")

        # Pass the user input to all agents
        ben_response = ben_agent.run(input=user_input)
        tyne_response = tyne_agent.run(input=user_input)
        da_response = da_agent.run(input=user_input)

        # Print responses from all agents
        print("Ben's Response:", ben_response)
        print("Tyne's Response:", tyne_response)
        print("Da's Response:", da_response)
        
if __name__ == '__main__':
    main()
