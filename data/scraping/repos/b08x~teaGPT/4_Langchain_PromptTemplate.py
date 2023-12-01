import os
import streamlit as st
import openai

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

try:
    if os.environ["OPENAI_API_KEY"]:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        openai.api_key = st.secrets.OPENAI_API_KEY
except Exception as e:
    st.write(e)


st.title("🦜🔗 Langchain - Blog Outline Generator App")


def blog_outline(topic):
    # Instantiate LLM model
    llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai.api_key)
    # Prompt
    template = "As an experienced data scientist and technical writer, generate an outline for a blog about {topic}."
    prompt = PromptTemplate(input_variables=["topic"], template=template)
    prompt_query = prompt.format(topic=topic)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    topic_text = st.text_input("Enter prompt:", "")
    submitted = st.form_submit_button("Submit")
    if not openai.api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        blog_outline(topic_text)
