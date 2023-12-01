import os
import streamlit as st
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

# os.environ["OPENAI_API_KEY"]

# App framework
st.title('Whatever GPT creator')
prompt = st.text_input("Plug in your prompt here")

# LLMS
llm = OpenAI(temperature=0.9)

if prompt:
    res = llm(prompt)
    st.write(res)
