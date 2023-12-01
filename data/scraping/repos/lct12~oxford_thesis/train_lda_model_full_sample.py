#!/usr/bin/env python
# coding: utf-8

# ### Extract Features from Tweets with the full sample

# #### Author: Lauren Thomas
# #### Created: 22/07/2021
# #### Last updated: 22/07/2021

# ###### File description: This file filters id2word_col and corpus_col from the full sample, excluding all words in fewer than 4 documents or more than 500 documents and tests the CVs of v
# Designed to be run on the server
# 

print('importing')
import os 
import pickle
import re
import spacy
import random
import time
import gzip
import gensim
import nltk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim.corpora as corpora
import seaborn as sns


from os import sep
from pprint import pprint
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

print('importing stopwords!')

# Import stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['RT'])

# Working directory
cwd = "/mnt/projects/lauren"
data_dir = "/mnt/projects/lauren"

# Path to mallet binary for the model generation
path_to_mallet_binary = f"{cwd}/mallet-2.0.8/bin/mallet"


# Create a function that trains LDA models using several diff num of topics, then pick the one with the best coherence

print('creating functions!')
def find_cv(num_topic_list, corpus, id2word, save_as, no_above_txt, no_below_txt, alpha=50, plot=True):
    '''num_topic_list = list of number of topics you want to run through/train'''

    #Create a dict that will have num of topics as key and model as value
    model_dict = dict()
    # Dict with coherence values
    coh_dict = dict()


    # Create model with various numbers of topics & the given alpha
    i = 0
    for num_topic in num_topic_list:
        print(f'modelling {num_topic} topics!')
        start = time.time()
        model = LdaMallet(path_to_mallet_binary, corpus=corpus, num_topics=num_topic, id2word=id2word, alpha=alpha)
        cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
        
        # Add model to model_dict
        model_dict[num_topic] = model
        
        coh_dict[num_topic] = cm.get_coherence()
        
        # Open text file info about the model
        model_output = open(f'{cwd}/output/model_output_{save_as}.txt', 'a')
        num_topics_txt = str(num_topic)
        # Append info about the model with n number of topics to file
        L = ['Model no.:', str(i), '\nNo. of topics:', str(num_topic), "\n",
             'No. below: ', no_below_txt, '; No. above: ', no_above_txt, "\n",
             'Coherence:',str(cm.get_coherence()),"\n",
            "Model topics: "]
        model_output.writelines(L)
        # Write each of the topics to the file
        for topic in model.print_topics(num_topic):
            model_output.writelines([str(topic), "\n"])
        model_output.writelines(['\n', '\n'])
        # Close the model
        model_output.close()
        print('Time Required:', round(time.time()-start,2), 'seconds')
        i += 1
        
        
        
    # Plot # of topics against coherence
    if plot==True:
        x,y = [num for num in coh_dict.keys()], [coh_value for coh_value in coh_dict.values()]
        sns.lineplot(x=x,y=y,marker='o')
        plt.title(f'Number of Topics vs. Coherence Scores for Alpha {alpha}')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
    
        plt.savefig(f'{cwd}/figures/{save_as}.jpg')
#         plt.show()
        
    return model_dict





### Filter out extreme words to improve CV


# Filter out extremes from the corpus using the dictionary id2word
# no_below = keep tokens that are in no more than ___ documents (abs number)
# no_above = keep tokens contained in no more than _% of total corpus (recall, only 20% of corpus is left) 
# keep_n = 
# Create a function that will filter collated tweets using no_below, no_above
def filter_corpus_cv(no_below, no_above, id2word_pickle_str, lemma_pickle_str, save_as, cv_list = [1,10,20,30,50]):
    # Unpickle id2word_col & lemmatized texts
    id2word_col = pickle.load(open(f'{cwd}/pickle/{id2word_pickle_str}.pickle', 'rb'))
    texts = pickle.load(open(f'{cwd}/pickle/{lemma_pickle_str}.pickle', 'rb'))
    
    print('Number of unique words!: ', len(id2word_col))
    
    # Filter id2word col using below & above
    id2word_col.filter_extremes(no_below=no_below, no_above=no_above)
    print('New number of unique words!:', len(id2word_col))
    
    # Create filtered corpus
    corpus_filtered = [id2word_col.doc2bow(text) for text in texts]
    
    # Create a (blank) text file that will contain info on coh, topics, etc.
    no_below_txt = str(no_below)
    no_above_txt = str(int(round(no_above*len(corpus_filtered), 0)))
    model_output = open(f'{cwd}/output/model_output_{save_as}.txt', 'w')
    L = ['No below: ', no_below_txt, "\n", "No above: ", no_above_txt, "\n", "\n"]
    model_output.writelines(L)
    model_output.close()
    
    # Run through various CV & alphas w/filtered dict & corpus. Return dictionary that contains model as values.
    model_dict_ab_bel = find_cv(cv_list, corpus_filtered, id2word_col,save_as, no_above_txt, no_below_txt)
    
    return model_dict_ab_bel


full_model_dict = dict()
full_cv_list = [1]
to_append = [5,10,15,20,25,30,35,40, 45, 50,55]
full_cv_list.extend(to_append)

# Run through vars in no_above_list and no_below_list, creating a full model nested dictionary that looks like the following:
# full_model_dict[(no_above, no_below)]

filter_tuple = (4, 500)
no_below_txt = "4"
no_above_txt = "500"
print('modelling topics for ' + str(filter_tuple))
save_as = "no_below_4_no_above_500_ym"
full_model_dict = filter_corpus_cv(4, 500/73354, "id2word_ym_all", "ym_collated_tweets_lemmatized_all", save_as, cv_list = full_cv_list)

# save_as = "no_below_4_no_above_500_year"
# full_model_dict_year = filter_corpus_cv(4, 500/51397, "id2word_col_year", "year_collated_tweets_lemmatized", save_as, cv_list = full_cv_list)

# Pickle full model dict
pickle.dump(full_model_dict, open(f'{cwd}/output/model_dict_ym_all.pickle', 'wb'))
# pickle.dump(full_model_dict_year, open(f'{cwd}/output/full_model_dict_year.pickle', 'wb'))










