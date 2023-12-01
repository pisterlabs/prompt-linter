from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.llms import OpenAIChat

os.environ["OPENAI_API_KEY"] = "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"

agent = create_csv_agent(
    OpenAI(temperature=0), "/home/capstone/Capstone/output.csv", verbose=True)

openaichat = OpenAIChat(model_name="gpt-3.5-turbo")

user_question = st.text_input("What do need to know about your network logs: ")

if user_question is not None and user_question != "":
    with st.spinner(text="In progress..."):
        st.write(agent.run(user_question))
        