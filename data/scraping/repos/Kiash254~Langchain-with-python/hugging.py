import streamlit as st
from langchain.llms import huggingface_endpoint

st.title('🦜🔗 Quickstart App')

endpoint = st.sidebar.text_input('Huggingface endpoint')

def generate_response(input_text):
    llm = huggingface_endpoint(endpoint=endpoint)
    st.info(llm(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if not endpoint.startswith('https://api-inference.huggingface.co/models/'):
        st.warning('Please enter your Huggingface endpoint!', icon='⚠')
    if submitted and endpoint.startswith('https://api-inference.huggingface.co/models/'):
        generate_response(text)
        
            