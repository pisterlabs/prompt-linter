import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

llm = OpenAI(openai_api_key="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")

st.set_page_config(page_title="Ask your CSV")
st.header("Ask your CSV 📈")

csv_file = st.file_uploader("Upload a CSV file", type="csv")
if csv_file is not None:

    # Create a CSV agent for your data file
    agent = create_csv_agent(OpenAI(temperature=0, openai_api_key='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'), csv_file, verbose=True)

    user_question = st.text_input("Ask a question about your CSV: ")

    if user_question is not None and user_question != "":
        with st.spinner(text="In progress..."):
            st.write(agent.run(user_question))