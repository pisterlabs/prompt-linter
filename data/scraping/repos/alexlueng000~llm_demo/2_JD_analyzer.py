import json

import streamlit as st 
from streamlit_pills import pills

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompts import CodeReviewerPromptTemplate, jd_system_template, jd_human_template

from langchain.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)



st.set_page_config(layout="wide", page_title="JD Analyzer")
st.title("JD Analyzer")

col1, col2 = st.columns([3, 2])

col2.markdown("### 分析结果")

col2.markdown("---")
col2.markdown("**推荐职位**")
container2 = col2.container()

col2.markdown("---") 

col2.markdown("**能力总结**")
container3 = col2.container()



def generate_response(input_text):
    print("Generating chat response...")
    system_message_prompt = SystemMessagePromptTemplate.from_template(jd_system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(jd_human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chat = ChatOpenAI(
        temperature=0.7, 
        openai_api_key=st.session_state['openai_api_key'],
        model="gpt-3.5-turbo-0613"
        )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(job_description=input_text)
    result = json.loads(response)
    keywords = result["关键字"]
    keyword_selected = pills(label="关键字", options=keywords, index=None)
    col1.write(keyword_selected)
    
    for job in result["推荐职位"]:
        container2.markdown("- " + job)
    
    container3.write(result["能力总结"])
    
    abilities = result["能力维度"]
    abilities_selected = pills(label="能力维度", options=abilities, index=None)
    col1.write(abilities_selected)


with col1.form('my_form'):
    text = st.text_area("Enter Job Description Here", height=400, placeholder="please enter the jd...")
    submitted = st.form_submit_button('开始分析')
    if not st.session_state['openai_api_key'].startswith("sk-"):
        print("openai key is not valid: ", st.session_state['openai_api_key'])
        st.warning("Please enter a valid OpenAI API Key!", icon='⚠')
    if submitted and st.session_state['openai_api_key'].startswith("sk-"):
        generate_response(text)

