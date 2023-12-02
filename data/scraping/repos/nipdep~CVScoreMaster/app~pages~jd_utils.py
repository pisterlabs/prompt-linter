import os
import math
import openai
import pandas as pd
import numpy as np
from dotenv import load_dotenv


from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from sklearn.metrics.pairwise import cosine_similarity

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from h2o_wave import Q

load_dotenv()
# extraction schemas

job_skill_schema = Object(
    id="skills",
    description="""
        A technical skill or programming language mentioned in a resume or a tool.
    """,
    attributes=[
        Text(
            id="skill",
            description="The name of a technical skill or programming language"
        )
    ],
    examples=[
        (
            "Experience in Python and Java",
            [
                {"skill": "Python"},
                {"skill": "nlp"},
                {"skill": "clustering"},
                {"skill": "classification"},
            ],
        ),
        (
            "Experience in machine learning",
            [
                {"skill": "data modeling"},
                {"skill": "Java"},
            ],
        ),
        (
            "Proficient in C++",
            [
                {"skill": "C++"}
            ]
        ),
        (
            "Familiarity with SQL databases",
            [
                {"skill": "SQL"}
            ]
        ), (
           "You must know AWS to do well in the job",
            [
               {"tool": "AWS"}
            ]
        ),
        (
            "Experience in web development using HTML, CSS, and JavaScript",
            [
                {"skill": "HTML"},
                {"skill": "CSS"},
                {"skill": "JavaScript"},
            ]
        )
    ],
    many=True,
)

jd_education_schema = Object(
    id="jd_education",
    description="""
        A Bachelor's Degree in Computer Science, Information Technology or equivalent
    """,
    attributes=[
        Text(
            id="jd_edu",
            description="A Bachelor's Degree in Computer Science, Information Technology or equivalent"
        )
    ],
    examples=[
        (
            "Degree in Computer Science, Software Engineering or Electronics / Electrical Engineering, or equivalent",
            {"jd_edu": "degree"},
        ),
        (
            "A degree in a relevant technical discipline or an equivalent professional qualification",
            {"jd_edu": "degree"},
        ),
        (
            "A Bachelor’s Degree in Computer Science, IT or an equivalent qualification",
            {"jd_edu": "degree"},
        ),
        (
            "Bachelors in computer science and engineering",
            {"jd_edu": "degree"}
        ),
        (
            "Degree in Computer Science, Software Engineering or Electronics / Electrical Engineering",
            {"ejd_edudu": "degree"}
        ),
        (
            "Bachelor's degree in Computer Science, Engineering, Mathematics or Statistics.",
            {"jd_edu": "degree"}
        ),
        (
            "BSc. Degree in Computer Science/ Information Technology",
            {"jd_edu": "degree"}
        ),
        (
            "Graduate in Computer Science or related field",
            {"jd_edu": "degree"}
        ),
        (
            "University degree in related subjects, equivalent training, or experience",
            {"jd_edu": "degree"}
        ),
        (
            "Masters in Data Science",
            {"jd_edu": "master"}
        ),
        (
            "Doctorate of Philosophy in computer vision",
            {"jd_edu": "phd"}
        )
    ],
    many=False,
)

jd_experience_schema = Object(
    id="jd_work_experience",
    description="""
        the work experience and job roles of the candidate
    """,
    attributes=[
        Text(
            id="internship",
            description="the internship experience of the candidate"
        ),
        Text(
            id="job_role",
            description="the job role the candidate has worked in"
        ),
        Number(
            id="years",
            description="the number of years of work experience"
        ),
        Number(
            id="months",
            description="the number of months of work experience"
        )
    ],
    examples=[
        (
            "1-2 years of experience in Software Development, Job Role: web developer, Duration: 1-2 years",
            {"job_role": "web developer", "years": 2}
        ),
        (
            "2-3 years’ experience in similar role, Job Role: Software Engineer, Duration: 2 years 3 months",
            {"job_role": "Software Engineer", "years": 2, "months": 3}
        ),
        (
            "Minimum 1+ years’ experience in the sphere of UI/UX, Job Role: UI/UX Engineer , Duration: 1 year",
            {"job_role": "UI/UX Engineer ", "years": 1, "years": 1}
        ),
        (
            "2+ year's of experience in testing planning, design, automation, and execution, Job Role: Quality Assurance Engineer, Duration: 2 year",
            {"job_role": "Quality Assurance Engineer", "years": 2}
        ),
        (
            "Minimum 1 year of experience in the software industry, Job Role: Quality Assurance Analyst , Duration: 1 year",
            {"job_role": "Quality Assurance Analyst ", "years": 1}
        ),
        (
            "Minimum 1 year of experience in the software industry, Job Role: Quality Assurance Engineer, Duration: 1 year",
            {"job_role": "Quality Assurance Engineer", "years": 1}
        ),
        (
            "2+ years of experience in manual testing web applications, Job Role: Quality Assurance Engineer, Duration: 2 years",
            {"job_role": "Quality Assurance Engineer", "years": 2}
        ),
        (
            "Minimum 3+ years experience in a software development role, Job Role: Software Engineer, Duration: 3 years",
            {"job_role": "Software Engineer", "years": 3}
        ),
        (
            "Have 2+ years of relevant work experience, Job Role: Data Analyst, Duration: 2 years",
            {"job_role": "Data Analyst", "years": 2}
        ),
    ],
    many=True,
)




def skill_extraction(chain, jd):
    try:
        return [i['skill'] for i in chain.predict_and_parse(text=jd)["data"]["skills"]]
    except:
        return []
    
def edu_extraction(chain, jd):
    try:
        edu = chain.predict_and_parse(text=jd)["data"]["education"][0]['edu']
        if edu == 'N/A':
            edu = np.nan
        return edu
    except:
        return np.nan

def exp_extraction(chain, jd):
    try:
        return sum([0 if d['years'] == '' else int(d['years'])*12+int(d['months']) for d in chain.predict_and_parse(text=jd)["data"]["work_experience"]])
    except:
        return np.nan
    
# extract main

def extract(q: Q, jd):
    # the extraction chain
    jd_skill_chain = create_extraction_chain(q.user.llm, job_skill_schema, input_formatter="triple_quotes")
    jd_edu_chain = create_extraction_chain(q.user.llm, jd_education_schema, input_formatter="triple_quotes")
    jd_exp_chain = create_extraction_chain(q.user.llm, jd_experience_schema, input_formatter="triple_quotes")

    data = {
        'skill': skill_extraction(jd_skill_chain, jd),
        'edu': edu_extraction(jd_edu_chain, jd),
        'exp': exp_extraction(jd_exp_chain, jd)
    }

    return data

# ========================================================================================
# JD based scoring 

def skill_score(jd, cv):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    cv_embs = openai.Embedding.create(
        input=cv,
        engine="text-similarity-davinci-001")

    jd_embs = openai.Embedding.create(
        input=jd,
        engine="text-similarity-davinci-001")

    mats = cosine_similarity(np.array([i['embedding'] for i in cv_embs['data']]), np.array([i['embedding'] for i in jd_embs['data']]))
    sim_score = mats.max(axis=1)
    sim_index = mats.argmax(axis=1)
    score = np.array([sim_score[np.where(sim_index == i)].mean() for i in range(mats.shape[1])])
    score[np.isnan(score)] = 0.0
    mean_score = score.mean()
    return mean_score

def edu_score(jd, cv):
    edu_map = {
        "after A/L": 0.1,
        "diploma": 0.2,
        "degree": 0.5,
        "master": 0.8,
        "phd": 1.0
    }
    edu_list = list(edu_map.keys())
    try:
        if isinstance(jd, float):
            score = edu_map[cv]
        else:
            jd_index = edu_list.index(jd)
            cv_index = edu_list.index(cv)
            score = 1.0 - abs(jd_index-cv_index)*0.2
    except:
        score = 0.0
    return score

def sin_transformation(x):
    scaled_x = (x/24)*np.pi/2
    result = (math.sin(scaled_x)+1.0)/2
    return result

def exp_score(jd, cv):
    cv_months = 0
    cv  = eval(cv)
    for elem in cv:
        try:
            j,n = elem.split(' | ')
        except:
            print("elem > ", elem)
        if (j!=""):
            if (n=="0"):
                cv_months+=6 
            else:
                cv_months += int(n)
        else:
            pass
        
    if not(np.isnan(jd)):
        month_count = cv_months-jd
    else:
        month_count = cv_months
    score = sin_transformation(month_count)
    return score

def total_score(r, skill=1, edu=1, exp=1):
    score = (skill*r['skill_score'] + edu*r['edu_score'] + exp*r['exp_score'])/3
    return score

# ==================================================================



