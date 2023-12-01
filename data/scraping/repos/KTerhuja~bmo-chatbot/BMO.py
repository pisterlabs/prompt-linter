import os
# from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from IPython.display import display, Markdown
import pandas as pd
import gradio as gr
import random
import time
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.agents.tools import Tool
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import OpenAI, VectorDBQA
from langchain.chains.router import MultiRetrievalQAChain
import streamlit as st
from streamlit_chat import message

# _ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size =1)
st.write("loading chroma")
bcar_retriever  = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/BCAR_Embedding").as_retriever()
smsb_retriever  = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/SMSB_EMBEDDING").as_retriever()
bmo_retriever  = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/BMO_FULL_EMBEDDING").as_retriever()
creditirb_retriever  = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/IRB").as_retriever()
creditstd_retriever  = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/credit_risk_standartize").as_retriever()
nbc_retriever  = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/NBC_Embedding").as_retriever()
st.write("loading qa")
qa_bcar = RetrievalQA.from_chain_type(llm=llm, retriever=bcar_retriever, verbose=True)
qa_bmo = RetrievalQA.from_chain_type(llm=llm, retriever=bmo_retriever, verbose=True)
qa_creditirb = RetrievalQA.from_chain_type(llm=llm, retriever=creditirb_retriever, verbose=True)
qa_creditstd = RetrievalQA.from_chain_type(llm=llm, retriever=creditstd_retriever, verbose=True)
qa_smsb = RetrievalQA.from_chain_type(llm=llm, retriever=smsb_retriever, verbose=True)
qa_nbc = RetrievalQA.from_chain_type(llm=llm, retriever=nbc_retriever, verbose=True)

tools = [
    Tool(
        name = "BCAR",
        func=qa_bcar.run,
        description="useful for when you need to find answer regarding bcar different categories and schedules"
    ),
    Tool(
        name="BMO Annual Report",
        func=qa_bmo.run,
        description="useful for when you need to find details about BMO bank like category it follows, fiscal year end etc"
    ),
    Tool(
        name="Credit Risk –Internal Ratings Based Approach",
        func=qa_creditirb.run,
        description="useful for when you need to find details about Credit Risk –Internal Ratings Based Approach "
    ),
    Tool(
        name="Credit Risk –Standardized Approach",
        func=qa_creditstd.run,
        description="useful for when you need to find details about Credit Risk –Standardized Approach "
    ),
    Tool(
        name="SMSB",
        func=qa_smsb.run,
        description="useful for when you need to find details about SMSB that is one category approach among BCAR"
    ),
    Tool(
        name="National Bnak Of Canada Annual Report",
        func=qa_nbc.run,
        description="useful for when you need to find details about National Bank of Canada like category it follows, fiscal year end etc"
    ),
]
planner = load_chat_planner(llm)

executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
#agent.run("Which reports Bank BMO has to send to OSFI for BCAR credit risk?")



## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
user_input = get_text()
if user_input:
  output = agent.run(user_input)
  st.session_state.past.append(user_input)
  st.session_state.generated.append(output)
if 'generated' in st.session_state:
    for i in range(len(st.session_state['generated'])-1,-1,-1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))