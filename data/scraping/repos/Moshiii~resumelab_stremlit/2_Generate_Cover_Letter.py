import streamlit as st
import PyPDF2
import os
import io
import time
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
import fitz
test = False


def read_jd(text_content):
    prompt_template = '''plrease reformat the following  text  to a Job description in the following format:
        resume_text:
        {text_content}
        Desired format: 
        Job Position:
            Position name
        Education qualification: 
            Degree and major
        Experience requirement: 
            Experience and number of years
        Programming Language: 
            list of Programming Languages
        Hard skill:
            list of Hard skill
        Soft skill:
            list of Soft skill
        Job respobsiability:
            summerized bullet points of responsiability
        Company Value:
            summerized company value and vision paragraph
        '''
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )

    result = llm_chain.predict(
        text_content=st.session_state['page_jd_jd_text_area'])
    return result

def generate_refined_resume():

    prompt_template = '''please generate a cover letter based on the Job description.
        Job description:
        {JD}
        '''
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )

    refined_resume = llm_chain.predict(
        JD=st.session_state['page_jd_JD'])
    return refined_resume

if 'page_jd_if_upload_clicked' not in st.session_state:
    st.session_state['page_jd_if_upload_clicked'] = False

if 'page_jd_if_resume_uploaded' not in st.session_state:
    st.session_state['page_jd_if_resume_uploaded'] = False

if 'page_jd_if_validate_clicked' not in st.session_state:
    st.session_state['page_jd_if_validate_clicked'] = False

if 'page_jd_if_generate_clicked' not in st.session_state:
    st.session_state['page_jd_if_generate_clicked'] = False

if 'page_jd_resume' not in st.session_state:
    st.session_state['page_jd_resume'] = ""

if 'page_jd_JD' not in st.session_state:
    st.session_state['page_jd_JD'] = ""

if 'page_jd_jd_text_area' not in st.session_state:
    st.session_state['page_jd_jd_text_area'] = ""

if 'page_jd_generated' not in st.session_state:
    st.session_state['page_jd_generated'] = False

if 'page_jd_refined_resume' not in st.session_state:
    st.session_state['page_jd_refined_resume'] = ""

if 'page_jd_text_content' not in st.session_state:
    st.session_state['page_jd_text_content'] = ""


st.markdown("Step 1. Provide your OpenAI API Key")
st.markdown("Step 2. Upload Job Description")
st.markdown("Step 3. Click 'Read JD.' AI will parse JD to text, and you may edit it before moving to the next step.")
st.markdown("Step 5. Click 'Generate Cover Letter.' AI will Generate Cover Letter based on the JD.")
st.markdown("Step 6. Click 'Download Cover Letter.' to save your result")

API_O = st.text_input('OPENAI_API_KEY', st.session_state['openAI_key'],type="password")
# API_O = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"
if API_O:
    llm = ChatOpenAI(temperature=0, openai_api_key=API_O,
                 model_name=MODEL, verbose=False)
else:
    st.info("please provide API Key")


jd_text_area = st.text_area('Upload JD', st.session_state['page_jd_JD'], 1000)

if st.button("Read JD"):
    if jd_text_area != "":
        st.session_state['page_jd_if_upload_clicked'] = True
    else:
        st.info("please make sure you provide all info")

if st.session_state['page_jd_if_upload_clicked'] == True:
    if st.session_state['page_jd_jd_text_area']=="":
        st.session_state['page_jd_jd_text_area'] = jd_text_area
        st.session_state['page_jd_if_resume_uploaded'] = True

if st.session_state['page_jd_if_resume_uploaded']:
    with st.spinner(text='Reading In progress'):
        if test:
            st.session_state['page_jd_resume'] = "test resume"
            st.session_state['page_jd_JD'] = "test JD"

        if st.session_state['page_jd_JD'] == "":
            jd_result = read_jd(st.session_state['page_jd_jd_text_area'])
            st.session_state['page_jd_JD'] = jd_result

        st.success('JD reading Completed')
        st.session_state['page_jd_JD'] = st.text_area(
            'JD', st.session_state['page_jd_JD'], 1000)
        if st.button("Generate Cover Letter"):
            st.session_state['page_jd_if_generate_clicked'] = True

if st.session_state['page_jd_if_generate_clicked']:
    with st.spinner(text='Optimize In progress'):
        if test:
            st.session_state['page_jd_refined_resume'] = "test refined_resume"
        if st.session_state['page_jd_refined_resume'] == "":
            refined_resume = generate_refined_resume()
            st.session_state['page_jd_refined_resume'] = refined_resume
        st.success('Resume Refined')
        st.session_state['page_jd_refined_resume'] = st.text_area(
            'cover letter', st.session_state['page_jd_refined_resume'], 1000)
        st.session_state['page_jd_generated'] = True
        st.download_button('Download Cover Letter', st.session_state['page_jd_refined_resume'],
                           file_name="Cover_Letter_ResumeLab")
