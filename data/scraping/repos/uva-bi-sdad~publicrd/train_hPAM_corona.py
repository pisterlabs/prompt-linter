import pandas as pd
import numpy as np
import pickle

import tomotopy as tp

import sys

import gensim
from gensim.models.coherencemodel import CoherenceModel

import time

from collections import Counter

import csv

import statistics

df = pd.read_pickle("../dspg20RnD/data/final/dashboard_data/corona_corpus.pkl")
docs = df["final_frqwds_removed"]

# function to get topic word distributions
def list_topics_hpam(mdl, top_n):
    
    topic_words = []
    
    for k in range(1 + mdl.k1 + mdl.k2):
        topic_words.append([words[0] for words in mdl.get_topic_words(k, top_n)])
    
    return topic_words

# hPAM topic dist
# function to get level of each topic as well as top 10 words in dist
def sub_topic_dist(mdl, top_n):
    
    sub_topics = []
    topic_words = []
    
    topic_words.append([-1, 0, mdl.get_topic_words(0, top_n = top_n)])
    
    for k in range(1, 1+mdl.k1):
        topic_words.append([0, k, mdl.get_topic_words(k, top_n = top_n)])
        
    for p in range(1+mdl.k1, 1+mdl.k1+mdl.k2):
        topic_words.append([1, p, mdl.get_topic_words(p, top_n = top_n)])
        
    topic_words_df = pd.DataFrame(topic_words)
    topic_words_df.columns = ['parent_level', 'topic_id', 'Top 10 words']
    
    for l in range(mdl.k1):
        subtopics = mdl.get_sub_topics(l, top_n = 3)
        sub_topics.append(subtopics)
      
    sub_topic_df = pd.DataFrame(sub_topics)

    return topic_words_df, sub_topic_df

def createTCvars(docs):
    
    # Create Dictionary
    id2word = gensim.corpora.Dictionary(docs)

    # Create Corpus (Term Document Frequency)

    #Creates a count for each unique word appearing in the document, where the word_id is substituted for the word
    corpus = [id2word.doc2bow(doc) for doc in docs]

    return id2word, corpus

def train_hpam(min_cf, rm_top, top_n, 
               alpha, eta, k1, k2,
               corpus, id2word, docs):
    
    # initialize hPAM                                                                            
    mdl = tp.HPAModel(tw = tp.TermWeight.IDF, min_cf = min_cf, rm_top = rm_top,
                      k1 = k1, k2 = k2, alpha = alpha, eta = eta, seed = 123)
    
    # load docs
    for abstracts in docs:
        mdl.add_doc(abstracts)
        
    # setup model                                                                                                     
    mdl.burn_in = 100
    mdl.train(0)

    # train model
    for i in range(0, 1000, 10):
        mdl.train(10)
        
    # create list of topics
    topics = list_topics_hpam(mdl, top_n = top_n)
    
    # calculate topic coherence
    cm = CoherenceModel(topics = topics, corpus = corpus, dictionary = id2word, 
                        texts = docs, coherence = 'c_v', processes = 8)
    
    cv = cm.get_coherence()
    
    # get topic distributions
    topic_words_df, sub_topic_df = sub_topic_dist(mdl, top_n = 10)
    
    # extract candidates for auto topic labeling
    extractor = tp.label.PMIExtractor(min_cf = min_cf, min_df = 0, max_len = 5, max_cand = 10000)
    cands = extractor.extract(mdl)

    # ranking the candidates of labels for a specific topic
    labeler = tp.label.FoRelevance(mdl, cands, min_df = 0, smoothing = 1e-2, mu = 0.25)

    label_lst = []
    for p in range(1+mdl.k1+mdl.k2):
        label_lst.append(label for label, score in labeler.get_topic_labels(p, top_n = 5))

    label_df = pd.DataFrame(label_lst)
    label_df.columns = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5']

    topic_words_label_df = pd.concat([topic_words_df.reset_index(drop = True), label_df], axis = 1)
    
    return cv, topic_words_label_df

docs_dict, docs_corpus = createTCvars(docs)

# hPAM parameters
alpha_vec = [0.15, 0.20, 0.25, 0.30, 0.35]
eta_vec = [0.3, 0.4, 0.5, 0.6, 0.7]
min_cf = 50
rm_top = 0
k1 = 7
k2 = 30

with open('hpam_results/corona/optimal_idf/hPAM_corona_all_tune.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['alpha', 'eta', 'min_cf', 'rm_top', 'k1', 'k2', 'tc'])
    
    for alpha in alpha_vec:
        for eta in eta_vec:
            
            tc_vec = []
            
            for l in range(10):
                
                tc, topic_labels = train_hpam(min_cf = min_cf,
                                              rm_top = rm_top,
                                              top_n = 10,
                                              alpha = alpha, 
                                              eta = eta, 
                                              k1 = k1,
                                              k2 = k2,
                                              corpus = docs_corpus, 
                                              id2word = docs_dict,
                                              docs = docs)
                tc_vec.append(tc)
                
            writer.writerow([alpha, eta, min_cf, rm_top, k1, k2, statistics.median(tc_vec)])

            #topic_labels.to_csv(r'hpam_results/corona/optimal_idf/hpam_corona_idf_alpha={}_eta={}_min_cf={}_rm_top={}_k1={}_k2={}.csv'.format(alpha, eta, min_cf, rm_top, k1, k2), index = False)
            
