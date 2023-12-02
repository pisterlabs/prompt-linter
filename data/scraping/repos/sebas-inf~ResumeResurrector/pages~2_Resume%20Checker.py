#Main File
import streamlit as st
import openai
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from streamlit_extras import add_vertical_space as avs
from langchain.callbacks import get_openai_callback
import os
import time

import requests
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from design import toggle 


st.set_page_config(page_title="Resume Reviewer", page_icon="📖")

toggle()

with st.sidebar:
    st.image("./images/panda_logo.png", caption="")
    st.title("Resume Resurrector")
    st.markdown('''
    This app is a LLM-powered resume reviewer built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                

 
    ''')

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

st.header("Resurrection Process")


# Set default job description and resume information
default_role = "Software Engineer"
default_jd  = "We are currently seeking a skilled and experienced Software Engineer to join our dynamic team. As a Software Engineer, you will play a crucial role in developing, designing, and maintaining software applications that meet the needs of our clients. You will collaborate with cross-functional teams to ensure the successful delivery of high-quality software solutions."
default_resume = "Resume: Personal Information: ..."

role_text = st.text_area("Hello User! Please provide the job role you'd like to apply for. Your resume will be judged based on this criteria", height=100, value=default_role)


# Enter job description
jd_text = st.text_area("Please provide the job description of the job you'd like to apply for.", height=100, value=default_jd)


pdf = st.file_uploader("Upload your resume for analysis!", type='pdf')
#st.write(pdf)


if pdf is not None:
    pdf_reader = PdfReader(pdf) 

       
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()

    prompt = f"""
You are a strict recruiter from top tier companies related to {role_text}. You have been assigned the task of evaluating a resume for a potential candidate with a background related to {role_text}.

Please carefully review the following resume and grade it on a scale of 0 to 100, with 0 being the minimum and 100 being the maximum score. Consider the criteria mentioned below while evaluating:

1. Education: Assess the candidate's academic qualifications and relevance to {role_text}. Look for degrees, courses, and educational institutions that are reputable in the field.

2. Technical Skills: Evaluate the candidate's proficiency in relevant frameworks, tools, and technologies commonly used in  relation to {role_text}. Look for specific skills mentioned in the resume.

3. Projects: Examine the candidate's previous projects, internships, or research work related to {role_text}. Assess the complexity, impact, and relevance of these projects.

4. Experience: Consider the candidate's professional experience in the related {role_text}, including internships, part-time jobs, or full-time positions. Evaluate the duration, responsibilities, and achievements in roles related to {role_text}.

5. Problem-Solving Abilities: Assess the candidate's problem-solving skills, logical reasoning, and ability to approach complex technical challenges. Look for any examples or indications of their problem-solving abilities.

6. Communication: Evaluate the candidate's written communication skills based on the quality and clarity of the resume itself. This includes formatting, organization, grammar, and overall presentation.

7. Additional Factors: Take into account any additional factors that you deem relevant for assessing the candidate's suitability for role.

Strictly display 5 of the least scored categories from above. Once you have reviewed the resume, please STRICTLY provide your final grade from a range within 0-100 based on the factors above and provide any specific feedback or comments to justify your evaluation. 
"""


    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a strict recruiter based on the following job description listed" + jd_text},
            {"role": "user", "content": resume_text + prompt}
        ]
    )

    progress_bar = st.progress(0)
    option_status = st.empty()

    response = chat['choices'][0]['message']['content']
    

    start_time = time.time()
    while (time.time() - start_time) < 2:
        # Update the progress bar
            
        progress = (time.time() - start_time) / 2
        progress_bar.progress(progress)

    
    st.write(response)
    progress_bar.empty()
    st.success("Done!")


    



# Parameter input



