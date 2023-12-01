import pandas as pd
import openai
import os
import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents.types import AGENT_TO_CLASS


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
}


def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


def get_qna_ans(df, question):
    pandas_df_agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors="Check your output and make sure it conforms!",
    )

    response = pandas_df_agent.run(question)
    st.markdown(response)
