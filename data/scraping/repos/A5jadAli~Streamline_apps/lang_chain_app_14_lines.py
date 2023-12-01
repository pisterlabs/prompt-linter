import streamlit as st
from langchain.llms import OpenAI

st.title("ChatGPT Clone")

# Input for OpenAI API Key
openai_api_key = st.sidebar.text_input('Enter OpenAI API Key', type='password')

# Function to generate response
def generate_response(input_text):
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter a valid OpenAI API key')
    elif not input_text:
        st.warning('Please enter some text')
    else:
        llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
        with st.spinner("Generating response..."):
            response = llm(input_text)
        st.info("Generated Response:")
        st.write(response)

# Streamlit form for user input
with st.form('my_form'):
    st.header("Enter Text")
    text = st.text_area("Input:", "What are the three pieces of advice for learning how to code?")
    submit_button = st.form_submit_button(label='Generate Response')

# Generate response when the form is submitted
if submit_button:
    generate_response(text)