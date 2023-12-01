import os
import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import *
from langchain.chains.summarize import load_summarize_chain
import tempfile
from langchain.docstore.document import Document
import time
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def load_cv(cv_file_path):
    loader = UnstructuredPDFLoader(cv_file_path, mode="elements", strategy="fast")
    docs = loader.load()
    return docs

def candidateAssessor(cv_str,job_description):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature=0
    )
    system_template = """You are an AI recruiter. Your task is to assess a candidate based on their CV and the job description.Please respond with Chinese"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please assess the following candidate based on their CV: '{cv_str}' and the job description: '{job_description}'."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(cv_str=cv_str, job_description=job_description)
    return result # returns string   

with st.form(key='HR_GPT'):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        placeholder="sk-...",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )

    st.title('HR AGENT')

    uploaded_file = st.file_uploader("上传CV", type=["pdf"], key='cv_file_path')

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            cv_file_path = temp_file.name
    else:
        cv_file_path = ''

    if cv_file_path:
        cv_doc = load_cv(cv_file_path)
    else:
        cv_doc = ''

    cv_str = "".join([doc.page_content for doc in cv_doc])

    job_description = st.text_area("输入job description")

    submit_button = st.form_submit_button(label='生成分析')

    if submit_button:
        if not openai_api_key.startswith('sk-'):
            st.warning('请输入有效的 OpenAI API key!', icon='⚠')
            assessment = ""
        elif cv_str and job_description:
            assessment = candidateAssessor(cv_str,job_description)
        else:
            assessment = ""

        if assessment:
            st.warning(assessment)
