# CONTAINS UNFISHED TESTING CODE


# Imports
import os
import warnings

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Pinecone imports
import pinecone
from langchain.vectorstores import Pinecone

# Open AI imports
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Chain imports
from langchain.chains.router import MultiRetrievalQAChain

# Agent imports
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.tools.render import format_tool_to_openai_function
# from langchain.agents.format_scratchpad import format_to_openai_functions
# from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain.agents import AgentExecutor

# Memory imports
# from langchain.memory.motorhead_memory import MotorheadMemory
from langchain.memory.buffer import ConversationBufferMemory

# Streamlit Import
# import streamlit as st

# Initialize pinecone and set index
pinecone.init(
    api_key= PINECONE_API_KEY,      
	environment='us-west4-gcp'      
)
index_name = "mojosolo-main"

# Initialize embeddings and AI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    temperature = 0.1,
    model_name="gpt-4"
)

# Initialize retrievers for MultiRetrievalQA Chain
client_retriever = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace="cust-projectwe-client-pinecone").as_retriever()
projectwe_retriever = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace="cust-projectwe-mojomosaic-pinecone").as_retriever()
muse_retriever = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace="cust-muse-mojomosaic-pinecone").as_retriever()
retriever_infos = [
    {
        "name": "client retriever", 
        "description": "Good for answering questions about people", 
        "retriever": client_retriever
    },
    {
        "name": "projectwe retriever", 
        "description": "Good for answering miscellaneous questions",
        "retriever": projectwe_retriever
    },
    {
        "name": "muse retriever", 
        "description": "Good for answering miscellaneous questions",
        "retriever": muse_retriever
    }
]

# Initialize Multiretrieval QA Chain and add to tools
qaChain = MultiRetrievalQAChain.from_retrievers(
    llm=llm, 
    retriever_infos=retriever_infos)  # Add verbose = True to see inner workings

# Custom tool function for upserting to pinecone
def upsertToPinecone(mem):
    print(mem)
    return "Saved " + mem + " to client database"

tools=[
    Tool.from_function(
        func=qaChain.run,
        name="Search pinecone",
        description="Useful for when you need to answer questions"
    ),
    Tool.from_function(
        func=lambda mem: upsertToPinecone(mem),
        name="Save to user database",
        description="Useful for when you need to save information to the user's database"
    )
]

# Memory (currently a Conversation Buffer Memory, will become Motorhead)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up Agent
agent_executor = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory = memory, verbose=True) # Add verbose = True to see inner workings
print("Enter your first query: ")
prompt = input()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    while(prompt.lower() != "quit"):
        print("MojoBob: ")
        print(agent_executor.run(prompt))
        print("Human: ")
        prompt = input()

# Contains last chat message from the AI
# memory.load_memory_variables({})["chat_history"][-1].content

# agent_executor.run("What is Henry's last name?")
# agent_executor.run("What is Conversations for Possibilities?")
# print(memory)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a chatbot having a conversation with a human."),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])
# llm_with_tools = llm.bind(
#     functions=[format_tool_to_openai_function(t) for t in tools]
# )
# agent = {
#     "input": lambda x: x["input"],
#     "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps'])
# } | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# agent_executor.invoke({"input": "What is Henry's last name?"})


# Motorhead Memory
# template = """You are a chatbot having a conversation with a human.

# {chat_history}
# Human: {human_input}
# AI:"""

# prompt = PromptTemplate(
#     input_variables=["chat_history", "human_input"], template=template
# )
# memory = MotorheadMemory(
#     session_id="testing-1", url="https://api.getmetal.io", memory_key="chat_history"
# )
# await memory.init




# Initialize docs and retrievers
# hen_docs = [
#     "Henry is an intern at Mojo Solo. His last name is Hoeglund. He likes computers."
# ]
# dav_docs = [
#     "David is the founder of MojoSolo. His last name is Matenaer. He likes working."
# ]



# If the index does not already exist, create it
# if(index_name not in pinecone.list_indexes()):
#     pinecone.create_index(name=index_name, metric='cosine', dimension=1536)