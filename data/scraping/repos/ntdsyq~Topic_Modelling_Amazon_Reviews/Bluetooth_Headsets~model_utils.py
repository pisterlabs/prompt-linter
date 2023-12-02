# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:55:56 2019

@author: yanqi
"""
import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 4 - Capstone\\AmazonReview\\Bluetooth_Headsets'
os.chdir(proj_path)

import pandas as pd
pd.set_option('display.max_colwidth', -1)  # to view entire text in any column
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import gzip
from pprint import pprint

import gensim
from gensim import corpora
from gensim.models import CoherenceModel

os.environ['MALLET_HOME'] = "C:/Users/yanqi/Library/mallet-2.0.8"
mallet_path = "C:/Users/yanqi/Library/mallet-2.0.8/bin/mallet"

import pyLDAvis
import pyLDAvis.gensim   # for visualizing found topics

#select_cat = 'Electronics->Computers & Accessories->Laptops'.split('->')
select_cat = 'Cell Phones & Accessories->Accessories->Headsets->Bluetooth Headsets'.split('->')
detail_cat = select_cat[-1].replace(' ','_')

def qc_dict(dictionary):
    # check some ids and tokens in the gensim dictionary
    # nt: number of terms
    nt = len(dictionary.keys())
    for i in random.sample(range(nt),10):
        print(i, dictionary[i])
        
def select_k(dictionary, corpus, texts, limit, start=3, step=2, method = 'mallet'):
    """
    Compute coherence for models with k number of topics to facilitate selecting the best model

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for k in range(start, limit, step):
        if method == 'mallet':
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=k, id2word=dictionary)
        else:  # the only other choice is gensim lda 
            LDA = gensim.models.ldamodel.LdaModel
            model = LDA(corpus=corpus, id2word=dictionary, num_topics=k, alpha = 'auto', eta = 'auto', passes = 10, iterations = 400, eval_every = 1, chunksize = 20)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print("finished training topic_number: ", k)

    return model_list, coherence_values
        
def out_doc_topics(ldamodel, corpus, rev_proc, rev_orig):
    # Init output
    doc_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Probability generated from this topic, and topic keywords 
        for j, (topic_num, prob_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num,5)
                topic_keywords = ", ".join([word for word, prob in wp])
                doc_topics_df = doc_topics_df.append(pd.Series([int(topic_num), round(prob_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    doc_topics_df.columns = ['Dominant_Topic', 'Prob_From_Topic', 'Topic_Keywords']

    # Add original text to the end of the output
    #contents = pd.Series(rev_proc)
    doc_topics_df = pd.concat([doc_topics_df, rev_proc, rev_orig], axis=1)
    return(doc_topics_df)

def out_topics_docs(ldamodel, corpus, prob0=0):
    # get a dictionary with topics as keys, and list of tuples (document_id, prob) as values
    # only keep tuples with prob > prob0
    topics_docs_dict = {}

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        # ldamodel[corpus][i] is a list of tuples (topic, prob) for document i
        for j, (topic_num, prob_topic) in enumerate(row):
            if prob_topic > prob0:
                if topic_num in topics_docs_dict:
                    topics_docs_dict[topic_num].append((i, prob_topic))  # add tuple (document_id, prob) 
                else: 
                    topics_docs_dict[topic_num] = [(i, prob_topic)]  # add the first document_id, prob pair
                
    return(topics_docs_dict)

def check_topic_doc_prob(topics_docs_dict,topic_num):
    # check the distribution of probabilities for a specified topic
    prob_lst = pd.Series([x[1] for x in topics_docs_dict[topic_num] ])
    return prob_lst

def topn_docs_by_topic(topics_docs_dict,topic_num, topn = 10):
    # return top 10 reviews and associated generating probability for specified topic
    top_docprobs = sorted(topics_docs_dict[topic_num], key = lambda x: x[1], reverse=True)[:topn]
    return top_docprobs

def make_bigrams(bigram_mod, texts):
    return [bigram_mod[doc] for doc in texts]

def load_processed_data():
    # load processed data
    df = pd.read_csv(detail_cat + "_processed.csv")
    reviews = df['review_lemmatized'].copy()
    reviews = reviews.apply(lambda x: x.split())
    return df, reviews