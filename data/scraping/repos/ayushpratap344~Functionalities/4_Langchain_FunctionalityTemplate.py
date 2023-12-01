import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

st.title("🦜🔗 Langchain - Prompt Template for Functionality")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


def blog_outline(topic):
    # Instantiate LLM model
    llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)
    # Prompt
    template = "As an experienced scientist, generate a 50-60 words passage on the functionality {topic} in field of cosmetics or paints or coatings in which it is most commonly used."
    prompt = PromptTemplate(input_variables=["topic"], template=template)
    prompt_query = prompt.format(topic=topic)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    topic_text = st.text_input("Enter functionality:", "")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        blog_outline(topic_text)
