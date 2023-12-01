name = "🖼️ Chooch Image Agent"
arguments = ["images"]
annotated = ["ZeroShot Agent","Default LLM","Path, Chooch Tool"]


from langchain.agents import ZeroShotAgent, AgentExecutor
from models.tools.chooch_chat import ChoochChatTool
from models.tools.image_path_finder import ImagePathFinderTool
from models.memory.custom_image_memory import CustomImageMemory
from models.memory.custom_image_memory import CustomImageMemory
from langchain import OpenAI, LLMChain

from models.llms.llms import *

def agent(images):
    tools = [ImagePathFinderTool(paths = images),ChoochChatTool()]
    prefix = """Have a conversation with a human, 
    Your role is to chat with an image.
    answering the following questions as best you can. You have access to the following tools:

    """
    suffix = """Begin!

    {chat_history}
    
    Question: {input}
    {agent_scratchpad}"""
    memory = CustomImageMemory(memory_key="chat_history",llm=chat_llm)
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    
    llm_chain = LLMChain(llm=OpenAI(temperature=0,streaming=True), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory,return_intermediate_steps=True
    )

    return agent_chain