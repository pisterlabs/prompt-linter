name = "💬 Summary Agent"
arguments = ["vectorstore"]
annotated = ["ZeroShot Agent","Default LLM","Path, Chooch Tool"]



from models.memory.custom_image_memory import CustomImageMemory

from models.tools.summarize_file import SummarizeFileTool
from models.tools.summarize_documents import SummarizeDocumentsTool
from langchain.agents import ZeroShotAgent, AgentExecutor

from langchain import OpenAI, LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor
from models.llms.llms import *
from models.tools.file_path_finder import FilePathFinderTool
import json
import streamlit as st
def agent(vectorstore):
    paths =json.loads(open(f'dataset/process/{st.session_state.user_id}/{st.session_state.session_id}/input/vector/metadata.json', 'r').read())
    tools = [
             FilePathFinderTool(paths = paths),
             SummarizeFileTool(vectorstore=vectorstore,paths = paths)
             ]
    prefix = """Have a conversation with a human, 
    answering the following questions as best you can. 
    You have access to the following tools:"""
    suffix = """Begin!"
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""
    memory = CustomImageMemory(memory_key="chat_history",llm=chat_llm)
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    
    llm_chain = LLMChain(llm=OpenAI(temperature=0,streaming=True), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory,return_intermediate_steps=True
    )
    return agent_chain
