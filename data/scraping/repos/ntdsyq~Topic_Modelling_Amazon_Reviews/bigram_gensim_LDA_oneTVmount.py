# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:22:18 2019

@author: yanqi
"""

import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 4 - Capstone\\AmazonReview'
os.chdir(proj_path)
import pandas as pd
pd.set_option('display.max_colwidth', -1)  # to view entire text in any column
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
import string
import gzip
from pprint import pprint

import gensim
from gensim import corpora
from gensim.models import CoherenceModel, LdaMulticore

import pyLDAvis
import pyLDAvis.gensim   # for visualizing found topics
from model_utils import qc_dict, out_topics_docs, check_topic_doc_prob, topn_docs_by_topic, select_k, make_bigrams

import warnings
warnings.simplefilter('ignore')  # suppressing deprecation warnings when running gensim LDA, 

# try: https://stackoverflow.com/questions/33572118/stop-jupyter-notebook-from-printing-warnings-status-updates-to-terminal?lq=1

import logging  # add filename='lda_model.log' for external log file, set level = logging.ERROR or logging.INFO
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG) 

## Load data, prepare corpus and dictionary

select_cat = 'Electronics->Accessories & Supplies->Audio & Video Accessories->TV Accessories & Parts->TV Ceiling & Wall Mounts'.split('->')

# useful attributes & methods: dictionary.token2id to get mapping, dictionary.num_docs 
df = pd.read_csv(select_cat[-1] + "_processed.csv", index_col= 0)
df = df[ df['asin'] == '0972683275']
df.reset_index(drop=True, inplace = True)
reviews = df['review_lemmatized'].copy()
reviews = reviews.apply(lambda x: x.split())

# Build the bigram models
bigram = gensim.models.Phrases(reviews, min_count=3, threshold=3) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)
print(bigram_mod[reviews[15]])

reviews_unigram = reviews.copy()
reviews = make_bigrams(bigram_mod, reviews)

# Dictionary expects a list of list (of tokens)
dictionary = corpora.Dictionary(reviews)
dictionary.filter_extremes(no_below=3)  # remove terms that appear in < 3 documents, memory use estimate: 8 bytes * num_terms * num_topics * 3

# number of terms
nd = dictionary.num_docs
nt = len(dictionary.keys())
print("number of documents", nd)
print("number of terms", nt)

qc_dict(dictionary)

# create document term matrix (corpus), it's a list of nd elements, nd = the number of documents
# each element of DTM (AKA corpus) is a list of tuples (int, int) representing (word_index, frequency)
DTM = [dictionary.doc2bow(doc) for doc in reviews]

# run lda model
LDA = gensim.models.ldamodel.LdaModel
#LDA = gensim.models.ldamulticore.LdaMulticore
n_topics = 10
passes = 10
iterations = 400

%time lda_model = LDA(corpus=DTM, id2word=dictionary, num_topics=n_topics, alpha = 'auto', eta = 'auto', passes = passes, iterations = iterations, eval_every = 1, chunksize = 20)
#%time lda_model = LDA(corpus=DTM, id2word=dictionary, num_topics=n_topics, eta = 'auto', passes = passes, iterations = iterations, eval_every = 1, workers = 3, chunksize = 2000)

## check priors, conherence score, and create topic visualization
coherence_lda_model = CoherenceModel(model=lda_model, texts=reviews, dictionary=dictionary, coherence='c_v')
cs = coherence_lda_model.get_coherence()
print("model coherence score is:", cs)

pprint(lda_model.print_topics())