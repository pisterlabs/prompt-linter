import streamlit as st
from langchain.llms import OpenAI

st.title('Fagfrokost ChatGPT Demo')

openai_api_key = st.sidebar.text_input('Legg inn OpenAI nøkler')

def generate_response(input_text):
  llm = OpenAI(temperature=0.1, openai_api_key=openai_api_key)
  st.info(llm(input_text))

with st.form('my_form'):
  text = st.text_area('Enter text:', 'Legg inn ditt spørsmål her')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)