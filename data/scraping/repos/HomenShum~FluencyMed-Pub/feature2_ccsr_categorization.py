import pandas as pd
import numpy as np
from numpy.linalg import norm
import openai
from feature1_clinical_note_summarization import patient_note_analysis_output
import streamlit as st

##### Datasets ############################################################################################################
ccsr_df_feather = pd.read_feather('DXCCSR-Reference-File-v2023-1.feather')

##### Settings ############################################################################################################
openai.api_key = st.secrets["openai_api_key"]

##### Functions ############################################################################################################
def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode() 
    response = openai.Embedding.create(input=content,engine=engine) 
    vector = response['data'][0]['embedding']  
    return vector

def similarity(v1, v2):  # return dot product of two vectors
    return np.dot(v1, v2)/(norm(v1)*norm(v2)) #dot product is a measure of similarity between two vectors

def search_index(text, data, count=5):
    vector = gpt3_embedding(text)  # get vector data for the question
    scores = []
    for i, content in zip(data, ccsr_df_feather['CCSR Category Description']):
        score = similarity(vector, i)  # compare vector data for the question versus each embedding
        scores.append({'content': content, 'score': score})  # create similarity scores for each document
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)  # in scores list, sort 0th "score" data -to- last "score" data by highest to lowest
    return ordered[:count]  # return top count documents

##### Main #################################################################################################################
input_text = patient_note_analysis_output

ccsr_categories_list = search_index(input_text, ccsr_df_feather['Embeddings'].tolist())

# print(ccsr_categories_list)
