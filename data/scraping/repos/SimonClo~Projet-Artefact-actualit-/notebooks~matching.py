# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Matching new articles

# %% [markdown]
# In this notebook we match new articles with a gensim model trained on a corpus of achives

# %% [markdown]
# ## Loading data

# %%
import os
import json
from nltk import word_tokenize
from stop_words import get_stop_words
import nltk
from nltk.corpus import stopwords

import pickle

import numpy as np

import gensim
from gensim.models import CoherenceModel
from gensim import corpora

import json

from nltk.stem.snowball import FrenchStemmer

import pandas as pd

stemmer = FrenchStemmer()

# %%
with open("./corpus2.pkl","rb") as f:
    corpus = pickle.load(f)
ldamodel = gensim.models.ldamodel.LdaModel.load('./model2.gensim')
dictionary = gensim.corpora.dictionary.Dictionary.load("./dictionary2.gensim")
with open("./scores.pkl","rb") as f:
    scores = pickle.load(f)

# %%
with open("./data/actualité/gilets_jaunes.json","r") as file :
    article = json.loads(file.read())[0]

# %% [markdown]
# ## Preprocessing article

# %%
stop_words = set(stopwords.words('french'))

# add stop words
stop_words_to_add = ['a', 'peut', 's', 'plus', 'si', 'tout', 'ce', 'cette', 'mais', 'être',
                     'c', 'comme', 'sans', 'aussi', 'fait', 'ça', 'an', 'sous', 'va', 'année', 'années', 'premier', 'premiers', 'première',
                     'vit', 'donner', 'donne', 'dernier', 'derniers', 'dernière', 'rien', 'reste', 'rester', 'bien', 'semain'
                    'autours', 'porte', 'prépare', 'préparer', 'trois', 'deux', 'quoi', 'quatre', 'cinq', 'six', 'sept', 'homme', 'jeune', 'france',
                    'entre', 'grand', 'grands', 'grande', 'grandes', 'après', 'partout', 'passe', 'jour', 'part', 'certains', 'certain',
                     'quelqu', 'aujourd', 'million', 'contre', 'pour', 'petit', 'ancien', 'demand', 'beaucoup', 'toujours'
                    'lorsqu', 'jusqu', 'hommme', 'seul', 'puis', 'faut', 'autr', 'toujour']
stop_words_to_add += get_stop_words('fr')

for word in stop_words_to_add:
    stop_words.add(word)

# %% [markdown]
# ### Removing stop words and punctuation

# %%
#removing punctuation : 
text = "".join([char if (char.isalnum() or char==" ") else " " for char in article["text"]])
text = word_tokenize(text)

# remove stop words
text = [word.lower() for word in text if word.lower() not in stop_words]

# %% [markdown]
# ### Lemmatization

# %%
text = [stemmer.stem(word) for word in text]
text = [word for word in text if len(word)>3]
print(text)

# %% [markdown]
# ## Scoring Article

# %% [markdown]
# ### Creating Bag Of Words representation

# %%
bow_article = dictionary.doc2bow((token) for token in text)

# %% [markdown]
# ### Scoring using the trained model

# %%
new_score = np.array([score[1] for score in ldamodel[bow_article][0]])
print(new_score)

# %%
topics2 = ldamodel2.print_topics(num_words=10)


for topic in topics2:
    print(topic)
    print()

# %% [markdown]
# ## Finding closest archive

# %%
score_matrix = np.array(scores)
print(score_matrix.shape)


# %% [markdown]
# ### Using Cosine similarity

# %% [markdown]
# For now scores are probability, we can get normalised vector for the euclidian space by applying the square root term by term.

# %%
def normalised_cosine_similarity(x,y):
    return np.sum(np.sqrt(x)*np.sqrt(y))

idx_min = 0
min_dist = 10
for i in range(score_matrix.shape[0]) : 
    dist = normalised_cosine_similarity(score_matrix[i],new_score)
    if dist < min_dist :
        min_dist = dist
        idx_min = i
        
print(idx_min)
print(min_dist)
print(score_matrix[idx_min])


# %%
def cosine_similarity(x,y):
    return np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y))


# %% [markdown]
# ### Using euclidian distance

# %%
def euclidian_distance(x,y):
    return np.linalg.norm(x-y)

idx_min = 0
min_dist = 10
for i in range(score_matrix.shape[0]) : 
    dist = np.linalg.norm(score_matrix[i]-new_score)
    if dist < min_dist :
        min_dist = dist
        idx_min = i
        
print(idx_min)
print(min_dist)
print(score_matrix[idx_min])


# %% [markdown]
# ### In the general case

# %%
def get_matching(score_matrix,new_score,distance,num_matches=5) :
    distances = np.zeros((score_matrix.shape[0]))
    for i in range(score_matrix.shape[0]) :
        distances[i] = distance(score_matrix[i],new_score)
        sorted_indexes = np.argsort(distances)
    return sorted_indexes[:num_matches]



# %%
print(get_matching(score_matrix,new_score,euclidian_distance))
