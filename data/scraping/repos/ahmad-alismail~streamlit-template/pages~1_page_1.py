import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.agents import load_tools, initialize_agent

# Set API keys from session state
openai_api_key = st.session_state.openai_api_key
serper_api_key = st.session_state.serper_api_key

# Streamlit app
st.subheader('Web Search')