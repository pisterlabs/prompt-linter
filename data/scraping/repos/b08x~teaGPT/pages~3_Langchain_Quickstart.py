import os
import streamlit as st
import openai


from langchain.llms import OpenAI

try:
    if os.environ["OPENAI_API_KEY"]:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        openai.api_key = st.secrets.OPENAI_API_KEY
except Exception as e:
    st.write(e)


st.title("🦜🔗 Langchain Quickstart App")

with st.sidebar:
    st.write("hey")


def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai.api_key)
    st.info(llm(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:", "What are 3 key advice for learning how to code?")
    submitted = st.form_submit_button("Submit")
    if not openai.api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        generate_response(text)
