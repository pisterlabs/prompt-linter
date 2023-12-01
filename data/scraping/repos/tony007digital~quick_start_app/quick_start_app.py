import os
import streamlit as st
from langchain.llms import OpenAI

st.title('🦜🔗 Quickstart App')

openai_api_key = os.getenv('OPENAI_API_KEY')

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    response = llm(input_text)
    print('Input text:', input_text)
    print('Generated response:', response)
    st.info(response)


with st.form('my_form'):
    text = st.text_area('Enter text:', 'Start typing here')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.lower().startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
