import openai
from openai.embeddings_utils import get_embedding, cosine_similarity # must pip install openai[embeddings]
import pandas as pd
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv



load_dotenv()




# set keys and configure Azure OpenAI
openai.api_type = 'azure'
openai.api_version = os.environ['AZURE_OPENAI_VERSION']
openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
openai.api_key = os.environ['AZURE_OPENAI_KEY']

# read in the embeddings .csv 
# convert elements in 'embedding' column back to numpy array
df = pd.read_csv('microsoft-earnings_embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

# caluculate user query embedding 
search_term = input("Enter a search term: ")
if search_term:
    search_term_vector = get_embedding(search_term, engine='text-embedding-ada-002')

    # find similiarity between query and vectors 
    df['similarities'] = df['embedding'].apply(lambda x:cosine_similarity(x, search_term_vector))
    df1 = df.sort_values("similarities", ascending=False).head(5)

    # output the response 
    print('\n')
    print('Answer: ', df1['text'].loc[df1.index[0]])
    print('\n')
    print('Similarity Score: ', df1['similarities'].loc[df1.index[0]]) 
    print('\n')