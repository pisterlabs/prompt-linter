import streamlit as st
st.set_page_config(layout="wide")
import streamlit.components.v1 as components
from streamlit_extras.stateful_button import button
from top_n_tool import run_tool
import os
from dotenv import load_dotenv
load_dotenv()
import markdown
import pandas as pd
import numpy as np
import time as time
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional
from io import StringIO
import langchain
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".st_langchain.db")
from datetime import datetime
import sqlite3
from sqlalchemy import create_engine
from llama_index import OpenAIEmbedding, ServiceContext, StorageContext,load_index_from_storage
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import HumanInputRun
from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.embeddings import OpenAIEmbeddings
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool

from streamlit_feedback import streamlit_feedback

# from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from custom_tools import ResearchPastQuestions

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.agents import create_pandas_dataframe_agent
from langchain.retrievers.llama_index import LlamaIndexRetriever
from langchain.chains import RetrievalQAWithSourcesChain


# Constants
DATA_PATH = "reddit_legal_cluster_test_results.parquet"


def clean_names(df):
    df.columns = [x.replace(' ', '_').lower() for x in df.columns]
    return df

 
@st.cache_data
def get_df():
    """Returns a pandas DataFrame."""
    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
    df['datestamp'] = df['timestamp'].dt.date
    # df['state'] = pd.Categorical(df['state'])
    df['text_label'] = pd.Categorical(df['text_label'])
    df['topic_title'] = pd.Categorical(df['topic_title'])
    return df


@st.cache_resource
def gen_profile_report(df, *report_args, **report_kwargs):
    return df.profile_report(*report_args, **report_kwargs)


def create_pandas_profiling():
    df = get_df()
    pr = gen_profile_report(df, explorative=True)
    with st.expander("REPORT", expanded=False):
        st_profile_report(pr)


def save_to_csv(data_list: list, file_name: str = "user_feedback.csv"):
    # Check if data_list is None or empty
    if not data_list:
        st.toast("⚠️ Warning: Nothing to save!")
        return
    
    # Convert list of dicts to DataFrame
    data_df = pd.DataFrame(data_list)
    # Add timestamp
    data_df['timestamp'] = datetime.now()

    # Check if file exists
    if Path(file_name).exists():
        # Append to existing csv
        data_df.to_csv(file_name, mode='a', header=False, index=False)
        st.toast(f"✔️ Feedback added to existing file:\n`{file_name}`")
    else:
        # Create a new csv and add rows
        data_df.to_csv(file_name, mode='w', header=True, index=False)
        st.toast(f"Created new file: `{file_name}`\n\n 👍 Data has been saved!")


def display_description():
    """Displays the description of the app."""
    st.markdown("<h4 style='text-align: left;'>Chat with an AI research agent</h4>", unsafe_allow_html=True)
    st.write(
        """
        Why use an AI agent?
        - 🧰 Agents have access to tools 
        - 🔧 Tools are small programs that do specific tasks
        - 👉 You give the instructions, and the agent figures out which tool, or set of tools to use
        """
    )


msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

if "feedback" not in st.session_state:
    st.session_state.feedback = []
    
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
    
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

reset_button = st.sidebar.button("Export Feedback")
if reset_button:
    save_to_csv(st.session_state.feedback)

with st.sidebar:
    view_messages = st.expander("Message History")
    view_feedback = st.expander("Feedback Log")

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("Hi! How can I help you?")
    st.session_state.steps = {}
    st.session_state.feedback = []


display_description() 

# create_pandas_profiling()


avatars = {"human": "https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/user_question_resized_pil.jpg", 
           "ai": "https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/c3po_icon_resized_pil.jpg"}

for idx, msg in enumerate(msgs.messages):
    with st.chat_message(msg.type, avatar=avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)


human_feedback_placeholder = st.empty()

def get_input() -> str:
    with human_feedback_placeholder.container():
        prompt = st.chat_input("human_input", key=f"human_{len(msgs.messages)}")
    return prompt


if prompt := st.chat_input(placeholder="Send a message"):
    st.chat_message("user", avatar="https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/user_question_resized_pil.jpg").write(prompt)
    st.session_state.last_prompt = str(prompt)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", verbose=True)
    df_ = get_df()
    df = clean_names(df_)
    human_tool = HumanInputRun(input_func=get_input)
    research_past_questions = ResearchPastQuestions(df=df)
    tools = load_tools(["ddg-search", "llm-math", "open-meteo-api"], llm=llm)
    tools.append(human_tool)
    tools.append(research_past_questions)
    
    
    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=ChatOpenAI(model_name="gpt-4", verbose=True), 
        tools=tools,
        return_source_documents=True,)
    
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_source_documents=True,
    )
    with st.chat_message("assistant", avatar="https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/c3po_icon_resized_pil.jpg"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True, collapse_completed_thoughts=False, max_thought_containers=10)
        response = executor(prompt, 
                            callbacks=[st_cb],
                            )
        st.write(response["output"])
        st.session_state.last_response = str(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]


if st.session_state["steps"]:
        feedback_ = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{len(msgs.messages)}",
        )
        if feedback_:
            feedback_['user_query'] = st.session_state.last_prompt
            feedback_['llm_response'] = st.session_state.last_response
            st.session_state.feedback.append(feedback_)
            time.sleep(3)
            st.toast("✔️ Feedback received!")


with view_messages:
    """
    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
    
with view_feedback:
    st.write(st.session_state.feedback)