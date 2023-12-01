from os import SCHED_OTHER
import streamlit as st
import pickle

# *******
import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)
# *********


c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.title('SMART GRADER')

col1, col2 = st.columns(2)

with col1:
    st.markdown("***")
    prof_ans = st.text_input(label = 'Professor Answer')
with col2:
    st.markdown("***")
    stud_ans = st.text_input(label = 'Student Answer')



dataset = load_dataset("trec", split="train")
df = pd.DataFrame(dataset)[:450]
api_key = 'AHKLrKZmFltl6RQHg4J2sNsqV4SS09YdEgLT31aL'

co = cohere.Client(api_key)
embeds = co.embed(texts=list(df['text']),
              model='large',
              truncate='LEFT').embeddings
search_index = AnnoyIndex(np.array(embeds).shape[1], 'angular')

for i in range(len(embeds)):
   search_index.add_item(i, embeds[i])
search_index.build(10) # 10 trees
search_index.save('test.ann')


    # *************
def predict(query):
    query_embed = co.embed(texts=[query],
                model="large",
                truncate="LEFT").embeddings
    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0],4,
                                                include_distances=True)
    # Format the results
    results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['text'], 
                            'distance': similar_item_ids[1]})

    #print(f"Query:'{query}'\nNearest neighbors:")
    return results


    # **********
prof_out = predict(prof_ans)
stud_out = predict(stud_ans)

print(prof_out)
def cal_score(prof_out, stud_out):
    diff = np.array(prof_out['distance']) - np.array(stud_out['distance'])
    diff_sum = np.array(diff).sum()
    score = (3.4*(np.log(6.1-6.4*diff_sum))+5)
    return score

score = cal_score(prof_out, stud_out)

if st.button('Predict Score'):
    with col1:
        st.write("Professor Answer : ", prof_ans)
        st.write(prof_out)

    with col2:
        st.write("Student Answer : ", stud_ans)
        st.write(stud_out)

    with col1:
        st.markdown('***')
        st.subheader("Total Score : ")
        st.markdown('***')

    with col2:
        st.markdown('***')
        st.subheader(score)
        
        st.markdown('***')
