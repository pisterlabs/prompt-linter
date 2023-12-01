import pandas as pd
import numpy as np
import pickle

import tomotopy as tp

import sys

import gensim
from gensim.models.coherencemodel import CoherenceModel

import time

from collections import Counter

df = df = pd.read_pickle("../dspg20RnD/data/final/final_dataset_7-20.pkl")
docs = df["final_frqwds_removed"]

def list_topics_hlda(mdl, top_n):
    
    topic_words = []
    topic_levels = []
    
    for k in range(mdl.k):
        if not mdl.is_live_topic(k):
            continue
        topic_words.append([words[0] for words in mdl.get_topic_words(k, top_n)])
        topic_levels.append(mdl.level(k))
    
    return topic_words, dict(Counter(topic_levels))

def createTCvars(docs):
    
    # Create Dictionary
    id2word = gensim.corpora.Dictionary(docs)

    #keep_only_most_common=int(len(docs)/2) #LDA works best with less features than documents
    #Filter words to only those found in at least a set number of documents (min_appearances)
    #id2word.filter_extremes(no_below=5, no_above=0.6, keep_n=keep_only_most_common)

    # Create Corpus (Term Document Frequency)

    #Creates a count for each unique word appearing in the document, where the word_id is substituted for the word
    corpus = [id2word.doc2bow(doc) for doc in docs]

    return id2word, corpus

def train_hlda(min_cf, rm_top, top_n, 
               alpha, eta, gamma, depth,
               corpus, id2word, docs):
    
    # initialize PA model                                                                                                                
    mdl = tp.HLDAModel(tw = tp.TermWeight.IDF, min_cf = min_cf, rm_top = rm_top,
                       depth = depth, alpha = alpha, eta = eta, gamma = gamma,
                       min_df = 20)
    
    # load docs
    for abstracts in docs:
        mdl.add_doc(abstracts)
        
    # setup model                                                                                                                          
    mdl.burn_in = 100
    mdl.train(0)

    # train model
    #print('Training...', file=sys.stderr, flush=True)
    for i in range(0, 1000, 10):
        mdl.train(10)
        #print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
        
    # create list of topics
    topics, level_count = list_topics_hlda(mdl, top_n = top_n)
    
    # calculate topic coherence
    cm = CoherenceModel(topics = topics, corpus = corpus, dictionary = id2word, 
                        texts = docs, coherence = 'c_v', processes = 10)
    
    cv = cm.get_coherence()
    
    return cv, level_count

docs_dict, docs_corpus = createTCvars(docs)

#alpha_vec = [0.005, 0.01, 0.025, 0.1]
alpha = 0.1
#eta_vec = [0.05, 0.1]
eta = 0.2
#gamma_vec = [0.01, 0.05, 0.1, 0.2]
gamma = 0.2
#min_cf_vec = [0, 1, 2]
min_cf = 2
#rm_top_vec = [5, 10, 15]
rm_top = 10
#depth_vec = [4, 5, 6, 7, 8]
depth = 4

param_tune_mat = []

tc, topics_in_lvl = train_hlda(min_cf = min_cf, 
                               rm_top = rm_top, 
                               top_n = 10,
                               alpha = alpha, 
                               eta = eta, 
                               gamma = gamma, 
                               depth = depth,
                               corpus = docs_corpus, 
                               id2word = docs_dict,
                               docs = docs)

param_tune_mat.append([alpha, eta, gamma, min_cf, rm_top, depth, tc, sum(topics_in_lvl.values())])

param_tune_df = pd.DataFrame(param_tune_mat)
param_tune_df.columns = ['alpha', 'eta', 'gamma', 'min_cf', 'rm_top', 'depth', 'tc', 'total_topics']
param_tune_df.to_csv(r'hlda_results/hlda_full_tune.csv', index = False)

