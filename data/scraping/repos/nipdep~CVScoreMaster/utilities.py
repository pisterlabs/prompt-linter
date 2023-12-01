
import os 
import sys 
import requests 
from dotenv import load_dotenv

import pandas as pd 
import numpy as np 

import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer,util
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from h2o_wave import Q

from .jd_utils import extract, skill_score, edu_score, exp_score, total_score

load_dotenv()

# Load primary data 

def load_model():
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    return emb_model

def load_llm():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=2000,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    return llm 

def load_js_data():
    df = pd.read_csv('./data/jds.csv')
    return df 

def load_cv_data():
    df = pd.read_csv('./data/cv_extractedV3.csv')
    return df 

def load_short_list():
    df = pd.read_csv('./data/shortlist.csv')
    return df

# save updated 

def save_df(df, db):
    if db == 'jd':
        df.to_csv('./data/jds.csv', index=False)
    elif db == 'cv':
        df.to_csv('./data/shortlist.csv', index=False)

# Process JD 

async def process_jd(q: Q, jd_info):
    extracted_data = extract(q, jd_info['content'])
    row_data = {
        'job': jd_info['job'],
        'content': jd_info['content'],
        'vacancies': jd_info['vacancies'],
        'skills': extracted_data['skill'],
        'education': extracted_data['edu'],
        'experience': extracted_data['exp']
    }
    temp_jds = q.user.jds.copy()
    temp_jds = temp_jds.append(row_data, ignore_index=True)
    q.user.jds = temp_jds

    save_df(temp_jds, 'jd')
    return row_data

# CVs Vs. JD similarity score 

def get_similarity_score(model, cvs, JD):
    cv_embs = model.encode(cvs)
    jd_emb = model.encode(JD)

    cos_sim = util.cos_sim(cv_embs, jd_emb)
    return cos_sim

# cv shortlisting 

async def shortlist_cvs(q: Q, JD_dict, k=10):
    cvs = q.user.cvs['content'] #@nipdep

    scores = get_similarity_score(q.user.emb_model, cvs, JD_dict['content'])[:, 0]
    top_cvs = np.array(list(np.argsort(scores)[-k:])[::-1])
    # print("top_cvs > ", top_cvs, ' scores >', scores)

    df = q.user.cvs.iloc[top_cvs, :]
    print("short listed df > ", df)

    df['target_job'] = JD_dict['job']
    df['similarity_score'] = scores[top_cvs]

    df['skill_score'] = df.apply(lambda r: skill_score(JD_dict['skills'], r['skills']) ,axis=1)
    df['edu_score'] = df.apply(lambda r: edu_score(JD_dict['education'], r['education']) ,axis=1)
    df['exp_score'] = df.apply(lambda r: exp_score(JD_dict['experience'], r['experience']) ,axis=1)
    df['total_score'] = df.apply(lambda r: total_score(r, skill=q.args.skill_weight, edu=q.args.edu_weight, exp=q.args.exp_weight) ,axis=1)
    return df 

async def update_short_list(q: Q, df):
    current_df = q.user.sl_cvs
    combined_df = pd.concat([current_df, df], axis=0)

    q.user.sl_cvs = combined_df
    save_df(combined_df, 'cv')

# JD Download & Processing

def read_text(path):
    full_string = ""
    with open(path, 'r') as pf:
        for line in pf.readlines():
            full_string += line 

    return full_string

async def download_jd(q: Q):
    local_path = await q.site.download(q.args.jd_file[0], '.')
    # print("local_path > ", local_path)
    content = read_text(local_path)

    jd_data = {
        'job': q.args.title,
        'vacancies': int(q.args.vacancies),
        'content': content
    }
    return jd_data 


# get current set of jobs

def get_jobs(q: Q):
    df = q.user.jds 
    jd_list = list(df['job'].values)
    vacancies = list(df['vacancies'].values)
    return jd_list, vacancies

# def job concent 

def get_job_content(q: Q, job_name):
    df = q.user.jds.copy()
    content = df.loc[df['job']==job_name, 'content'].values[0]
    return content

def get_job_shortlist(q: Q, job_name):
    df = q.user.sl_cvs.copy()
    job_shortlist = df.loc[df['target_job']==job_name, :]
    q.user.job_sl = job_shortlist
    return job_shortlist.loc[:, ['id', 'name', 'similarity_score']].values.tolist()

def get_job_detail(q: Q, job_name):
    df = q.user.jds.copy()
    content = df.loc[df['job']==job_name, :].to_dict(orient='records')[0]
    return content

def get_cv_details(q: Q, cv_id):
    df = q.user.job_sl.copy()
    # print("shortlist > ", df, "cv_id > ", cv_id)
    content = df.loc[df['id']==int(cv_id), :].to_dict(orient='records')[0]
    return content