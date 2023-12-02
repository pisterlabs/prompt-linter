import streamlit as st
from langchain import OpenAI

from utils import get_openai_api_key

openai_api_key = get_openai_api_key()


def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))


st.markdown("# LLM ChatUI")
st.sidebar.markdown("# LLM ChatUI")


with st.form('my_form'):
    text = st.text_area(
        'Prompt:', '',
        placeholder=('What are the ways generative AI can '
                     'enhance my job as a Software Engineer?')
    )
    submitted = st.form_submit_button('Submit')
    if submitted and text:
        generate_response(text)
