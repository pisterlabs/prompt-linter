# Implement demo code by Streamlit: https://blog.streamlit.io/langchain-tutorial-1-build-an-llm-powered-app-in-18-lines-of-code/

import streamlit as st
from langchain.llms import OpenAI

# Load settings
openai_api_key = st.secrets["openai"]["key"]

# Page Contents
st.title('🦜🔗 Basic LangChain Example')
st.caption("🚀 Interact with OpenAI's LLMs via LangChain")

# Helper
def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
  st.info(llm(input_text))

# Execute
with st.form('my_form'):
  text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)



