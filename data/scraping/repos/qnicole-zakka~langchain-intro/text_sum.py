import streamlit as st
from langchain.llms import OpenAI
st.title("Test App")
my_key = st.sidebar.text_input("OpenAI API Key")

def generate_response(input_text):
    llm = OpenAI(temperature=0.5, openai_api_key=my_key)
    st.info(llm(input_text))


with st.form('my_form'):
    text = st.text_area('Enter text here: ', 'Chinese law bar v.s. U.S. law bar, which test is harder')
    submitted = st.form_submit_button('Submit')
    if not my_key.startswith("sk-"):
        st.warning("Please enter a valid OpenAI API Key")
    if submitted and my_key.startswith("sk-"):
        generate_response(text)
import streamlit 
