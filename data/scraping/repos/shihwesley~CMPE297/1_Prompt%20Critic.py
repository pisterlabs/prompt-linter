import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

st.title("💬 Prompt Critic")

openai_api_key = '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'


def prompt_critic(prompts):
    # Instantiate LLM model
    llm = OpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
    # Prompt
    template = "As an expert prompt enigneer, critique this '{prompts}'. Provide feedback and areas of improvement. At the end, please improve it"
    prompt = PromptTemplate(input_variables=["prompts"], template=template)
    prompt_query = prompt.format(prompts=prompts)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    topic_text = st.text_input("Enter prompt:", "")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        prompt_critic(topic_text)
