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

def read_resume(text_content):
    prompt_template = '''plrease reformat the following  text  to a resume in the following format:
        resume_text:
        {text_content}
        Desired format: 
        Summary:
            personal summary
        Skills: 
            list of skill limited to 10
        Experience: 
            company, role
            details
            company, role
            details
            ...
        Projects: 
            project name (skill list)
            details
            project name (skill list)
            details
            ...
        Eduation:
            university name and major | start time - end time
            university name and major | start time - end time
            ...
        '''
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )

    result = llm_chain.predict(text_content=st.session_state['text_content'])
    return result


def get_suggestion():

    prompt_template = '''plrease list 6 short one sentence suggestions to improve the resume. 3 suggestions are given already, please make sure to include them in the output. 
        
        for example :
            Try to add numbers and metrics in the Experience and Projects to make it more impressive
            Try to include technical skill keywords in bullet points 
        resume_text:
        {text_content}
        Suggestions(please include the following three):
        1.Polish text and Fix all grammar issue. 
        2.Try to add numbers and metrics in the Experience and Projects to make it more impressive
        3.Try to include technical skill keywords in bullet points
        '''
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )

    suggestions = llm_chain.predict(
        text_content=st.session_state['resume'])
    return suggestions


def generate_refined_resume():

    prompt_template = '''plrease polish the following resume based on the suggestions.
        suggestions:
        {suggestions}
        resume_text:
        {text_content}
        '''
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )

    refined_resume = llm_chain.predict(
        suggestions=st.session_state['suggestions'], text_content=st.session_state['resume'])
    return refined_resume


    
if 'openAI_key' not in st.session_state:
    st.session_state['openAI_key'] = ""

if 'if_upload_clicked' not in st.session_state:
    st.session_state['if_upload_clicked'] = False

if 'if_resume_uploaded' not in st.session_state:
    st.session_state['if_resume_uploaded'] = False

if 'if_validate_clicked' not in st.session_state:
    st.session_state['if_validate_clicked'] = False

if 'if_generate_clicked' not in st.session_state:
    st.session_state['if_generate_clicked'] = False

if 'resume' not in st.session_state:
    st.session_state['resume'] = ""

if 'suggestions' not in st.session_state:
    st.session_state['suggestions'] = ""

if 'generated' not in st.session_state:
    st.session_state['generated'] = False

if 'refined_resume' not in st.session_state:
    st.session_state['refined_resume'] = ""

if 'text_content' not in st.session_state:
    st.session_state['text_content'] = ""


st.markdown("Step 1. Provide your OpenAI API Key")
st.markdown("Step 2. Upload your resume")
st.markdown("Step 3. Click 'Read Resume.' AI will parse your resume to text, and you may edit it before moving to the next step.")
st.markdown("Step 4. Click 'Make Suggestions.' AI will provide you with suggestions for polishing your resume.")
st.markdown("Step 5. Click 'Auto Improve.' AI will polish your resume based on the suggestions.")
st.markdown("Step 6. Click 'Download Resume.' to save your result")


API_O = st.text_input('OPENAI_API_KEY', st.session_state['openAI_key'],type="password")
# API_O = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"
if API_O:
    st.session_state['openAI_key'] = API_O
    llm = ChatOpenAI(temperature=0, openai_api_key=API_O,
                 model_name=MODEL, verbose=False)
else:
    st.info("please provide API Key")




uploaded_file = st.file_uploader("Choose a file", type="pdf")

if st.button("Read Resume"):
    if uploaded_file is not None and API_O:
        st.session_state['if_upload_clicked'] = True
    else:
        st.info("please make sure you provide all info")
if st.session_state['if_upload_clicked'] == True:
    if st.session_state['text_content']=="":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = os.path.join("tempDir", uploaded_file.name)

        doc = fitz.open(pdf_path)
        text_content = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += page.get_text()

        st.session_state['text_content'] = text_content
        st.session_state['if_resume_uploaded'] = True

if st.session_state['if_resume_uploaded'] == True:
    with st.spinner(text='Reading In progress'):
        if test:
            st.session_state['resume'] = "test resume"

        if st.session_state['resume'] == "":
            result = read_resume(st.session_state['text_content'])
            st.session_state['resume'] = result
        st.success('Resume reading Completed')
        st.session_state['resume'] = st.text_area('Resume', st.session_state['resume'], 1000)
        if st.button("Make suggestions"):
            st.session_state['if_validate_clicked'] = True

if st.session_state['if_validate_clicked']:
    with st.spinner(text='validating In progress'):
        if test:
            st.session_state['suggestions'] = "test suggestions"

        if st.session_state['suggestions'] == "":
            suggestions = get_suggestion()
            st.session_state['suggestions'] = suggestions
        st.info('Suggestions')
        st.write(st.session_state['suggestions'])
        if st.button("Auto Improve"):
            st.session_state['if_generate_clicked'] = True

if st.session_state['if_generate_clicked']:
    with st.spinner(text='Polish In progress'):
        if test:
            st.session_state['refined_resume'] = "test refined_resume"

        if st.session_state['refined_resume'] == "":
            refined_resume = generate_refined_resume()
            st.session_state['refined_resume'] = refined_resume
        st.success('Resume Refined')
        st.session_state['refined_resume'] = st.text_area('Resume', st.session_state['refined_resume'], 1000)
        st.session_state['generated'] = True
        st.download_button('Download Resume', st.session_state['refined_resume'],
                           file_name="Polished_resume_ResumeLab")
        
