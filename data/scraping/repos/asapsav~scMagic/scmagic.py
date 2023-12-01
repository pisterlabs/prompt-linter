#This code is a fork from Langchain and Streamlit's Agent repo:
#https://github.com/langchain-ai/streamlit-agent/tree/main
#Credit to these amazing open source developers

import streamlit as st

import pandas as pd

import os
from dotenv import load_dotenv
load_dotenv()


from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import create_python_agent , create_pandas_dataframe_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

#from callbacks.capturing_callback_handler import playback_callbacks
from clear_results import with_clear_container

from prompts import PLANNER_AGENT_MINDSET

st.set_page_config(
    page_title="scMagic", page_icon="🧬🦜", layout="wide", initial_sidebar_state="collapsed"
)
st.title("🧬 scMagic: run scRNA-seq analysis")

# Setup credentials in Streamlit
user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Set this to run your own custom questions."
)
user_openai_api_key = os.environ.get('OPENAI_API_KEY')

st.session_state.disabled = True
if user_openai_api_key:
    st.session_state.disabled = False


#def book_charpter_12():
    # parse book charpter into context of claude 100K and ask for code
    #pass

tools = [
    #Tool(
    #    name="Normalise the data",
    #    func=book_charpter_12,
    #    description="Use when ....",
    #)
]

# Initialize agent
agent_executor = create_python_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=user_openai_api_key, streaming = True),
    tool=PythonREPLTool(), 
    # how does REPL tool work? 
    # we need to make it work so it remebmers results of all REPLs before (as if it were living in jupyter notebook)
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    agent_executor_kwargs={"handle_parsing_errors": True},
)
from sklearn.datasets import load_iris
data = load_iris(as_frame=True).data
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=user_openai_api_key),
    data,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

with st.form(key="form"):
    user_input = st.text_input("Formulate your scRNA-seq research question")
    submit_clicked = st.form_submit_button("Submit Question", disabled=st.session_state.disabled)

output_container = st.empty()
if with_clear_container(submit_clicked) and user_openai_api_key:
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🦜")
    st_callback = StreamlitCallbackHandler(answer_container)


    answer = agent_executor.run(user_input, callbacks=[st_callback])

    answer_container.write(answer)