from langchain.memory.buffer import ConversationBufferMemory
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
import streamlit as st
import os
import warnings
#Initialize Openai
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv('.env')
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Initialize embeddings and AI
projectwe_embeddings = OpenAIEmbeddings()

# Prompt Template
prompt = PromptTemplate(
    input_variables=['question','context'],
    template='Write an answer for this: {question}\n{context}\nAnswer:'
)
# Llms
llm = ChatOpenAI(temperature=0)
projectwe_chain = LLMChain(llm=llm, prompt=prompt)

# Initialize pinecone and set index
pinecone.init(
    api_key=PINECONE_API_KEY,      
	environment='us-west4-gcp'      
)
index_name = "mojosolo-main"
#FOR PROJECTWE
projectwe_retriever = Pinecone.from_existing_index(index_name=index_name, embedding=projectwe_embeddings, namespace="cust-projectwe-mojomosaic-pinecone")
#projectwe memory
projectwe_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#ProjectWe-Chain
chain1 = load_qa_chain(llm, memory=projectwe_memory, chain_type="map_reduce")

p_chain = ConversationalRetrievalChain(
    retriever=projectwe_retriever.as_retriever(),
    question_generator=projectwe_chain,
    combine_docs_chain=chain1,
)

#For MUSE
# Initialize embeddings and AI
muse_embeddings = OpenAIEmbeddings()
# Prompt Template
prompt = PromptTemplate(
    input_variables=['question','context'],
    template='Write an answer for this: {question}\n{context}\nAnswer:'
)
# Llms
llm = ChatOpenAI(temperature=0.5)
muse_chain = LLMChain(llm=llm, prompt=prompt)

muse_retriever = Pinecone.from_existing_index(index_name=index_name, embedding=muse_embeddings, namespace="cust-muse-mojomosaic-pinecone")
#projectwe memory
muse_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#ProjectWe-Chain
chain2 = load_qa_chain(llm, memory=muse_memory, chain_type="map_reduce")

m_chain = ConversationalRetrievalChain(
    retriever=muse_retriever.as_retriever(),
    question_generator=muse_chain,
    combine_docs_chain=chain2,
)

#initilaize tool
tools=[
    Tool.from_function(
        func=m_chain.run,
        name="Search muse pinecone namespace",
        description="Generate sections from Muse Bootstrap snippets for projectwe"
    ),
    Tool.from_function(
        func=p_chain.run,
        name="Search projectwe pinecone namespace",
        description="Useful for when you need to answer miscellaneous questions"
    )
]

# Set up Agent
model_name="gpt-3.5-turbo-16k-0613"
agent_llm = ChatOpenAI(temperature=0.8,model_name=model_name, max_tokens=8192)
#Agent memory
agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = initialize_agent(tools, agent_llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, memory = agent_memory,verbose=True)
print("Enter your first query: ")
prompt = input()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    while(prompt.lower() != "quit"):
        print("MojoBob: ")
        print(agent_executor.run(prompt))
        print("Human: ")
        prompt = input()