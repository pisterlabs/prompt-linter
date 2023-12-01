import os
# from dotenv import load_dotenv, find_dotenv
# from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
# from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
# from langchain.llms import AzureOpenAI
# from langchain.document_loaders import DirectoryLoader,PyPDFLoader
# from langchain.document_loaders import UnstructuredExcelLoader
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
# from IPython.display import display, Markdown
# import pandas as pd
# import gradio as gr
# from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
# from langchain.vectorstores import Chroma
# from langchain.agents.tools import Tool
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain import OpenAI, VectorDBQA
# from langchain.chains.router import MultiRetrievalQAChain
import streamlit as st
from streamlit_chat import message
# from langchain.document_loaders import UnstructuredPDFLoader
# _ = load_dotenv(find_dotenv())

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1)


template = """
You are virtual assistant of OSFI.
Use the following  context (delimited by <ctx></ctx>), and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(input_variables=["history", "context", "question"],template=template)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size =1)

bcar_retriever = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='Basel Capital Adequacy Reporting (BCAR) 2023 (2)_index')
bmo_retriver = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='bmo_ar2022 (2)_index')
creditirb_retriever = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='Capital Adequacy Requirements (CAR) Chapter 5 Credit Risk Internal Ratings Based Approach_index')
creditstd_retriever = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='Capital Adequacy Requirements (CAR) Chapter 4  Credit Risk Standardized Approach_index')
nbc_retriever = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='NATIONAL BANK OF CANADA_ 2022 Annual Report (1)_index')
smsb_retriever = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='SMSB (1)_index')

indices = [bcar_retriever,bmo_retriver,creditirb_retriever,creditstd_retriever,nbc_retriever,smsb_retriever]

for index in indices[1:]:
    indices[0].merge_from(index)



agent = RetrievalQA.from_chain_type(llm = llm,
    chain_type='stuff', # 'stuff', 'map_reduce', 'refine', 'map_rerank'
    retriever=bcar_retriever.as_retriever(),
    verbose=False,
    chain_type_kwargs={
    "verbose":True,
    "prompt": prompt,
    "memory": ConversationBufferMemory(
        memory_key="history",
        input_key="question"),
})

# st.title("BMO Chatbot")

# if 'something' not in st.session_state:
#     user_input = ''

# def submit():
#     user_input = st.session_state.widget
#     st.session_state.widget = ''

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []
# ## past stores User's questions
# if 'past' not in st.session_state:
#     st.session_state['past'] = []
# messages = st.container()
# user_input = st.text_input("Query", key="widget", on_change=submit)
# relevent_docs = st.expander("Relevent Docs", expanded=False)
# if user_input:
#     output = agent.run(user_input)
#     with relevent_docs:
#         st.write("\n\n\n",bcar_retriever.as_retriever().get_relevant_documents(user_input),"\n\n\n")
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)
# if 'generated' in st.session_state:
#     with messages:
#         for i in range(len(st.session_state['generated'])):
#             message(st.session_state['past'][i], is_user=True, key=str(i) + '_user',avatar_style="initials",seed="U")
#             message(st.session_state["generated"][i], key=str(i),avatar_style="initials",seed="A")

# agent.run("Which reports bank BMO has to send to OSFI for BCAR Credit Risk?")

# print(bcar_retriever.as_retriever().get_relevant_documents("what is fiscal year end of BMO?"))

st.title("BMO Chatbot")

# if 'something' not in st.session_state:
#     user_input = ''

# def submit():
#     user_input = st.session_state.widget
#     st.session_state.widget = ''

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []
messages = st.container()
user_input = st.chat_input("Query")
relevent_docs = st.expander("Relevent Docs", expanded=False)
if user_input:
    output = agent.run(user_input)
    # with relevent_docs:
    #     st.write("\n\n\n",bcar_retriever.as_retriever().get_relevant_documents(user_input),"\n\n\n")
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
if 'generated' in st.session_state:
    with messages:
        for i in range(len(st.session_state['generated'])):
            st.chat_message("user").write(st.session_state['past'][i])
            st.chat_message("assistant").write(st.session_state["generated"][i])
