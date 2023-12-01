from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import Counter
import google.generativeai as palm
import os
import json
import re

from dotenv import load_dotenv
load_dotenv()



def load_pdfs(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    pdf_content = "\n".join([page.page_content for page in pages])
    return pdf_content

def get_chunks(text, chunk_size=300, chunk_overlap=20):
    r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\.", "\n"]
    )
    split_data = r_splitter.split_text(text)
    return split_data


def get_projects(resume_content):
    #### Extract Projects from resume
    template = '''
    From the resume below extract the projects and their descriptions.
    extract two type of projects: personal and professional 
    Output should be in a dictionary format:
    example output:
    {
        "personal": [
            {
                "name": "project name here",
                "description": "description here"
            }
        ],
        "professional": [
            {
                "name": "project name here",
                "description": "description here"
            }
        ]
    }
    '''
    palm.configure(api_key=os.getenv("PALM_API_KEY"))
    response = palm.chat(
        context=template,
        messages=resume_content)

    return response.last

def get_skills_resume(resume_content):
    #### Extract Projects from resume
    template = '''
    From the resume below extract data science related skills
    Output should be like a python list

    example output: ['python', 'sql', 'machine learning', 'deep learning', 'data science']
    '''
    palm.configure(api_key=os.getenv("PALM_API_KEY"))
    response = palm.chat(
        context=template,
        messages=resume_content)

    return response.last

def get_skills_job_desc(job_desc):
    #### Extract Projects from resume
    template = '''
    From the job_description below extract data science related skills
    Output should be like a python list

    example output: ['python', 'sql', 'machine learning', 'deep learning', 'data science']
    '''
    palm.configure(api_key=os.getenv("PALM_API_KEY"))
    response = palm.chat(
        context=template,
        messages=job_desc)

    return response.last

def get_experience(resume_content):
    #### Extract Projects from resume
    template = '''
    From the resume below extract data science related experience
    Output should be a integer value denoting the number of months of experience

    example output: "24"
    '''
    palm.configure(api_key=os.getenv("PALM_API_KEY"))
    response = palm.chat(
        context=template,
        messages=resume_content)

    return response.last



def get_skills_regex(resume_content_chunks: list) -> list:
    """
    Load skills list from config.json and check for matches in the resume
    it might be better to use an LLM for this too.
    """
    with open("config.json") as f:
        data = json.load(f)
        entities = data["skills"]  # Check if skills is the correct key

    # Combine the skills for regex pattern
    pattern = "|".join(entities)
# Processing all files
    for file in resume_content_chunks:
        # Find all skills in the text
        skills.extend(re.findall(pattern, file, re.IGNORECASE))
    
    skills = list(set([skill.lower() for skill in skills]))

    # Get the unique values and their counts
    skills = Counter(skills)
    return skills
    # print(f"For file, found skills: {skills}") # for debugging