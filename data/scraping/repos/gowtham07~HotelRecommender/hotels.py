import requests
import os
import json
from time import time

import umap
import torch
import cohere
import warnings
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import torch.nn.functional as F

from typing import List, Union, Dict, Any


##Important functions

##get embeddings from cohere

model_name = 'multilingual-22-12' #@param ["multilingual-22-12", "small", "large"]
def get_embeddings(co: cohere.Client, model_name: str, texts: List[str], truncate: str = "RIGHT"):
    output = co.embed(model=model_name, texts=texts, truncate=truncate)
    return output.embeddings

## similarity function 
torchfy = lambda x: torch.as_tensor(x, dtype=torch.float32)

def get_similarity(target: List[float], candidates: List[float], top_k: int):
    candidates = torchfy(candidates).transpose(0, 1) # shape (768, bs)
    target = torchfy(target) # shape (1, 768)
    dot_scores = torch.mm(target, candidates)

    scores, indices = torch.topk(dot_scores, k=top_k)
    similarity_hits = [{'id': idx, 'score': score} for idx, score in zip(indices[0].tolist(), scores[0].tolist())]

    return similarity_hits   



url = "https://hotels4.p.rapidapi.com/locations/v3/search"

API_KEY = '5fa5658111mshd1008bbe356bc06p1ac1f6jsn9e45212f9333'
HEADERS = {
    'X-RapidAPI-Key': API_KEY,
    'X-RapidAPI-Host': 'hotels4.p.rapidapi.com',
}
COHERE_API_KEY = 'd8eTHtyzVN2e6LKLpy8E8xZkyFfmSwZWIayDhKIt'  #@param {type:"raw"}
co = cohere.Client(COHERE_API_KEY)

##Read the dataframe to display the city and location of hotel


df = pd.read_pickle("dummy.pkl")  
df['reviews.text'] = df['reviews.text'] + " Hotel is in "+ df['city'] +' has postalcode of ' + df['postalCode']

#load the embeddings

embeddings = torch.load('embeddings_kaggle.pt')
embeddings = embeddings.tolist()


# def search(query: str):
#     params = {'q': f'{query} hotels', 'locale': 'en_US'}
#     response = requests.request(
#         'GET', url, headers=HEADERS, params=params)
#     data = response.json()
#     result = []
#     for entity in data.get('sr', []):
#         if entity['type'] == 'HOTEL':
#             result.append(entity['regionNames']['displayName'])
#     return result

def search(query: str):
    sims = []
    top_k: int = 5 #@param {type:"slider", min:1, max:100, step:5}
    embeddings3 = embeddings.copy()
    query_embeddings = get_embeddings(co=co, model_name=model_name, texts=[query])
    similarity_hits = get_similarity(target=query_embeddings, candidates=embeddings3, top_k=top_k)
    sims.append(similarity_hits)
    ## the below three lines of code is useful if we accumulate two to three questions before we give an answer
    ## for now not sure how to make that work
    flat_list_sim = [item for sublist in sims for item in sublist]
    newlist = sorted(flat_list_sim, key=lambda d: d['score'],reverse=True)
    newlist = newlist[:5]
    similarity = [x['id'] for x in newlist]
    ##get reviews 
    review_list = []
    for i in range(len(similarity)):
       review_list.append(df.iloc[similarity[i]]['reviews.text']+ " The hotel name is "+ df.iloc[similarity[i]]['name']) 

    return review_list
