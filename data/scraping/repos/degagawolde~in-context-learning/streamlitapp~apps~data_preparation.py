# import packages
from cohere.classify import Example
from re import I
import missingno as msno
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
import streamlit as st

import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.append('./scripts')
print('ppppppp',os.getcwd())
from preprocess import Preporcess
from prompt_generation import cohereExtractor
from prompt_preprocess import TestPreprocess

prompt_prp = TestPreprocess()
prp = Preporcess()



def data_preparation(df: pd.DataFrame):

    df = pd.read_csv('./data/news/news_data.csv')
    st.write('# Prompt Design')
    st.write('## Prompt Desigin for News Scoring')
    st.write('Get the news data')
    st.dataframe(df)
    st.write('### Generate prompt from the dataframe')
    dpart = st.selectbox(
        'Select Document Part', ('Title', 'Description', 'Body'))
    df['Analyst_Rank'] = df['Analyst_Rank'].apply(lambda x: 0 if x < 4 else 1)
    
    examples = list()
    for txt, lbl in zip(df[dpart], df['Analyst_Rank']):
        examples.append(Example(txt, lbl))
    for ex in examples[5:7]:
        st.write(str(ex).replace('{', '\n').replace(
            '\t', "\n").replace('}', '\n').replace('cohere.Example',""))

    st.write('## Prompt Design for Entity Extraction')
    
    df = pd.read_json(
            './data/entity/relations_training.txt')

    st.dataframe(df)
    st.write('### Generate prompt from job description')
    df['label'] = df['tokens'].apply(prp.preprocess_tokens)
    train_doc = prp.preprocess_document(
        df, './data/output/training_prompt.txt')

    st.write(train_doc[0])
    

    
