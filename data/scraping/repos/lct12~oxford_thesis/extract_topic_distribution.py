#!/usr/bin/env python
# coding: utf-8

# ### Extract Features from Tweets with the full sample

# #### Author: Lauren Thomas
# #### Created: 02/08/2021
# #### Last updated: 02/08/2021

# ###### File description: This file extracts the topic distribution
###### for each document in the full corpus
# Designed to be run on the server
# 

print('importing')
import os 
import pickle

import gensim


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim.corpora as corpora


from os import sep
from pprint import pprint
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

print('unpickling corpus and model!')

# Working directory
cwd = "/mnt/projects/lauren"
data_dir = "/mnt/projects/lauren"

# Path to mallet binary for the model generation
path_to_mallet_binary = f"{cwd}/mallet-2.0.8/bin/mallet"

full_model_dict = pickle.load(open(f'{cwd}{sep}output{sep}model_dict_ym_all.pickle', 'rb'))

num_topics = 45
ldaMallet = full_model_dict[num_topics]

id2word = pickle.load(open(f'{cwd}{sep}pickle{sep}id2word_ym_all.pickle', 'rb'))
texts = pickle.load(open(f'{cwd}{sep}pickle{sep}ym_collated_tweets_lemmatized_all.pickle', 'rb'))

# Filter corpus
id2word.filter_extremes(no_below=4, no_above=500/73354)
print('New number of unique words!:', len(id2word))
    
# Create filtered corpus
corpus_filtered = [id2word.doc2bow(text) for text in texts]


print('assigning topic distribution')
topic_distribution = ldaMallet[corpus_filtered]

# Extract the topic distribution for each document to make a list of lists (list of each document, and
# within that, a list of distributions for each topic)
# topic_distribution[i][j][0] is tuple number for topic number j in document i
# topic_distribution[i][j][1] is distribution amount for topic number j in document i

doc_distribution_list = []
for i in range(len(topic_distribution)):
    distribution_list = []
    for j in range(num_topics):
        distribution_list.append(topic_distribution[i][j][1])
    doc_distribution_list.append(distribution_list)

print('pickling topic distribution')

pickle.dump(topic_distribution, open(f'{cwd}{sep}output{sep}topic_distribution_tuples.pickle', 'wb'))
pickle.dump(doc_distribution_list, open(f'{cwd}{sep}output{sep}topic_distribution_list.pickle', 'wb'))



