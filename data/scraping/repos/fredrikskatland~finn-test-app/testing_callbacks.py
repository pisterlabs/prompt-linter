import streamlit as st

from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client

import pinecone
from langchain.vectorstores import Pinecone
import os

client = Client()

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat🦜🔗"

local = False



@st.cache_resource(ttl="1h")
def configure_retriever():
    if local:
        pinecone.init(
            api_key=os.getenv("PINECONE_API_FINN"),  # find at app.pinecone.io
            environment = os.getenv("PINECONE_ENV_FINN")  # next to api key in console
        )
        embeddings = OpenAIEmbeddings() 
    else:
        pinecone.init(
            api_key=st.secrets["PINECONE_API_FINN"], 
            environment = st.secrets["PINECONE_ENV_FINN"]
        )
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])

    index_name = "finn-demo-app"
    vectorstore = Pinecone.from_existing_index(index_name = index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    return retriever




tool = create_retriever_tool(
    configure_retriever(),
    "search_finn_adds",
    "Searches and returns job adds from Finn.no. Finn.no is the leading online marketplace in Norway, everything from job ads to property and cars. You have access to a large selection of ads/listings, so you should use this tool when asked about jobs and positions.",
)
tools = [tool]

if local:
    llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4", )
else:
    llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4", openai_api_key=st.secrets["openai_api_key"])

message = SystemMessage(
    content=(
        "You are a helpful chatbot who is tasked with answering questions about job ads. "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about job ads. "
        "If there is any ambiguity, you probably assume they are about that."
    )
)
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)
memory = AgentTokenBufferMemory(llm=llm)
starter_message = "Ask me about open positions!"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]


def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id
