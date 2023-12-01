import os
import openai
from langchain.agents import initialize_agent, load_tools,Tool
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.document_loaders import GoogleDriveLoader
from langchain.agents import initialize_agent, AgentType
import sys
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import ast
from langchain.utilities import SerpAPIWrapper

llm = ChatOpenAI(
    temperature=0,
)
llm1 = OpenAI(temperature=0)

def chatAgent(tool, query):
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)
    zapier = ZapierNLAWrapper()
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    tools = toolkit.get_tools()

    tool = tool.strip()

    #convert sting to a list
    tool = ast.literal_eval(tool)
    new_tools = []
    tools_ids = []

    #get the tool which are suggested by the chain and strip them if there are unnecessay white spaces
    for i in range(len(tool)):
        tool[i] = tool[i].strip()

    #get the specific tools from zapierNLA
    for i in range(len(tool)):
        id = next(({'action_id': a.action_id, 'name' : a.name, 'description': a.description} for a in tools if a.name == tool[i]), None)
        if id is not None:
            tools_ids.append(id)

    #add those tools into a list
    for item in tools_ids:
        new_tools.append(
            Tool(
                name=item['name'],
                func=ZapierNLAWrapper().run(),
                description=item['description'],
                action_id=item['action_id']
            ))
        
    agent = initialize_agent(tools=new_tools, llm=llm, memory=memory, agent="chat-conversational-react-description", verbose=True, max_iteration=3, early_stopping_method='generate')
    
    
    fixed_prompt = '''
    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to sending emails, sceduling meetings with clients, and adding events to the calender.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''

    agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

    result = agent.run(query)

    return result


prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    What tool should be used for this: {query}?
    list of available tools are:["Gmail: Send Email", "Gmail: Find Email", "Gmail: Create Draft", "Google Calender: Find Event", "Google Calender: Quick Add Event", "Google Meet: Scedule a Meeting", "Slack: Add Remider", "Slack: Send Direct Message"]
    You can output multiple tools and they must be in python list. Don't return anything else. Only a python list. If you cannot find the tool for the task then return empty list            
    """,
)

chain = LLMChain(llm=llm1, prompt=prompt)

@cl.on_message  
def main(message: str):
    #get the specific tools from the LLM chain
    tool = chain.run(message)
    #pass those tools to the agent
    result = chatAgent(tool, query=message)
    cl.Message(content=result).send()
