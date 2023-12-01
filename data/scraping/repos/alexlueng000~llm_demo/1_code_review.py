import streamlit as st 
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompts import CodeReviewerPromptTemplate, cr_system_template, cr_human_template

from langchain.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)



reviewer = CodeReviewerPromptTemplate(input_variables=["source_code"], template="")


st.set_page_config(layout="wide", page_title="Code Reviewer")
st.title("Code Reviewer")

col1, col2 = st.columns([3, 2])


def generate_response(input_text):
    print("Generating chat response...")
    system_message_prompt = SystemMessagePromptTemplate.from_template(cr_system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(cr_human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chat = ChatOpenAI(
        temperature=0.7, 
        openai_api_key=st.session_state['openai_api_key'],
        model="gpt-3.5-turbo-0613"
        )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(source_code=input_text)
    print("chatgpt response: ", response)
    col2.write(response)
    # print(code_prompt)



with col1.form('my_form'):
    text = st.text_area("Enter code here: ", height=700, placeholder="Enter code here...")
    submitted = st.form_submit_button('review code')
    if not st.session_state['openai_api_key'].startswith("sk-"):
        print("openai key is not valid: ", st.session_state['openai_api_key'])
        st.warning("Please enter a valid OpenAI API Key!", icon='⚠')
    if submitted and st.session_state['openai_api_key'].startswith("sk-"):
        generate_response(text)
