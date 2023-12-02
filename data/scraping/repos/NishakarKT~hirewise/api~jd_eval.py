# JD Sections:
## 01. Job Title --> 10
## 02. Job Summary --> 5
## 03. Responsibilities --> 10
## 04. Requirements --> 10
## 05. Preferred Qualification --> 10
## 06. Company Overview --> 8 
## 07. Benefits & Perks --> 7
## 08. Salary Range --> 6
## 09. Location --> 5
## 10. Contact Information --> 5

## Section Completion metric: 
#  00-30: Red
#  30-60: Yellow
# 60-100: Green

# Readability Metric:
# Linsear Write Formula: (easy words{<=2 syllables} * 1 + hard words{>=3 syllables} * 3) / total words

from tika import parser
import os
import openai
from langchain.chains import SimpleSequentialChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
import textstat
import fastapi
from fastapi.middleware.cors import CORSMiddleware
from flask import jsonify
from pydantic import BaseModel
from fastapi.responses import JSONResponse
app = fastapi.FastAPI()  

origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Read the JD
# parsed_jd = parser.from_file('model_jd_2.txt')
# jd_contents = parsed_jd['content'].strip()

class JDInput(BaseModel):
    jd_contents: str

class JDResult(BaseModel):
    overall_readability_score: float
    section_scores: dict
    final_score: float
    recommended_sections: dict

@app.post("/jd_eval/")
def jd_eval(request: JDInput):
    jd_contents = request.jd_contents
    print(jd_contents)
    #Define the Prompt Template
    template = PromptTemplate.from_template(
        """You are a human resource manager. You are reading a job description for a job opening at your company. Given the job description, you need to answer the following questions:
        {job_description}
        Human: {question}
        AI:
        """)



    #Extract Section Headings from the JD
    required_sections = ['Job Title', 'Job Summary', 'Responsibilities', 'Requirements', 'Preferred Qualification', 'Company Overview', 'Benefits/Perks', 'Salary Range', 'Location', 'Contact Information']

    section_wts = {'Job Title':10, 'Job Summary':5, 'Responsibilities':10, 'Requirements':10, 'Preferred Qualification':10, 'Company Overview':8, 'Benefits/Perks':7, 'Salary Range':6, 'Location':5, 'Contact Information':5}

    llm_chain = LLMChain(
        llm=OpenAI(openai_api_key="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"),
        prompt=template,
        verbose=True,
    )

    result = llm_chain.predict(job_description=jd_contents, question="Which of the following sections are present in the job description? ['Job Title', 'Job Summary', 'Responsibilities', 'Requirements', 'Preferred Qualification', 'Company Overview', 'Benefits/Perks', 'Salary Range', 'Location', 'Contact Information']. Please just list the sections. Remember to put a comma between each section.")

    sections = result.split(',')
    sections = [section.strip(' ') for section in sections]
    sections = [section.strip("'") for section in sections]
    sections = [section for section in sections if section in required_sections]

    section_scores = [section_wts[section] for section in sections]
    final_score = sum(section_scores)/sum(section_wts.values())

    # print(final_score)

    # Extract the Section Contents
    section_contents = dict()
    for section in sections:
        section_text = llm_chain.predict(job_description=jd_contents, question=f"Extract and return the {section} of the job description? Return the section verbatim.")
        section_contents[section] = section_text

    # Calculate Readability Scores for each section
    section_scores = dict()
    for section in sections:
        readability_score = textstat.text_standard(section_contents[section], float_output=True)
        readability_score = readability_score - 12 if readability_score > 12 else 1
        section_scores[section] = readability_score


    # Calculate the overall Readability Score
    overall_readability_score = textstat.text_standard(jd_contents, float_output=True)
    overall_readability_score = overall_readability_score - 12 if overall_readability_score > 12 else 1

    # Recommed the sections to be rewritten
    recommended_sections = dict()

    for section in sections:
        section_content = section_contents[section]
        suggested_text = llm_chain.predict(job_description=section_content, question=f"Given the contents of the section {section}, please suggest a better way to write the section which will make it more readable.")
        recommended_sections[section] = suggested_text

    return_dict = {
        'overall_readability_score': overall_readability_score,
        'section_scores': section_scores,
        'final_score': final_score,
        'recommended_sections': recommended_sections
    }

    return (JSONResponse(return_dict))