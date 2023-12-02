import validators, streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# Set API keys from session state
openai_api_key = st.session_state.openai_api_key