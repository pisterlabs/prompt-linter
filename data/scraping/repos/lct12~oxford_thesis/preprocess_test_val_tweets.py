#!/usr/bin/env python
# coding: utf-8

# ### Extract Features from Tweets with the full sample

# #### Author: Lauren Thomas
# #### Created: 29/07/2021
# #### Last updated: 29/07/2021

# ###### File description: This file preprocesses the validation and/or test samples
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
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# Up max length
nlp.max_length = 2000000

# Working directory
cwd = "/mnt/projects/lauren"
data_dir = "/mnt/projects/lauren"

# Path to mallet binary for the model generation
path_to_mallet_binary = f"{cwd}/mallet-2.0.8/bin/mallet"

# Preprocess full sample
print('unpickling test/val sample')
ym_collated_tweet_list_test_val = pickle.load(open(f'{data_dir}{sep}pickle{sep}ym_collated_tweet_list_val_test.pickle', 'rb'))


print("Preprocessing full sample!")

# Generate functions that will be used in the preprocessing of the tweets

# Create function that checks for any word that begins with certain chars in string passed then returns string without those words
def check_for_word(check_list, string):
    # Check list = list of strings that we do not want to begin any word in our passed string (e.g. http, @, [pic])
    for check in check_list:
        if check in string:
            # Split splits string on whitespace to words, filter gets rid of any word beginning w/the relevant chars, join joins them again with whitespace in between
            string = " ".join(filter(lambda x:x[0:len(check)]!=check, string.split()))
    return string

# Create function to clean up each tweet (get rid of emojis, URLs, etc.)
def sent_to_words(tweet_list):
    for tweet in tweet_list:
        # Check for URLs or [pic] (indicating picture?) or @ or any stopword-- if it exists, delete them.
        tweet = check_for_word(['http', '[pic]', '@'], tweet)
        # Remove new line characters
        tweet = re.sub('\n', ' ', tweet)
        yield(gensim.utils.simple_preprocess(str(tweet), deacc=True))
        
# Make function to remove stopwords 
def remove_stopwords(tweet_words):
    return [[word for word in tweet if word not in stop_words] for tweet in tweet_words]

# Make functions to lemmatize data
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Now, create function that preprocesses the data given a list, where every item in the list = a document
# and the entire list = a corpus. Returns the dictionary & corpus
def preprocess_tweet_data(tweet_list, lemma_pickle):
    print('starting preprocessing...')
    # Clean up tweets
    tweet_words = list(sent_to_words(tweet_list))
    print('tweets cleaned!')
    
    # Remove stopwords
    tweets_nostops = remove_stopwords(tweet_words)
    print('stopwords removed!')
    
    # Create lemmatized data
    tweets_lemmatized = lemmatization(tweets_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print('tweets are lemmatized!')
    
    # Create corpus for use in topic modeller
    
    texts = tweets_lemmatized
    
    # Pickle the lemmatized tweets
    pickle.dump(texts, open(f'{data_dir}{sep}pickle{sep}{lemma_pickle}.pickle', 'wb'))
    return texts
     

# # Preprocess test/val data
print('preprocessing YM tweet data')
texts_val_test = preprocess_tweet_data(ym_collated_tweet_list_test_val, "ym_collated_tweets_lemmatized_test_val")


# Now, combine the lemmatized tweets in train with those in test and val
tweets_all = pickle.load(open(f'{data_dir}{sep}pickle{sep}ym_collated_tweets_lemmatized.pickle', 'rb'))
tweets_all.extend(texts_val_test)

# Dump new combined lemmatized texts
pickle.dump(tweets_all, open(f'{data_dir}{sep}pickle{sep}ym_collated_tweets_lemmatized_all.pickle', 'wb'))


# Now, create new dictionary with tweets_all
id2word = corpora.Dictionary(tweets_all)
texts = tweets_all

corpus = [id2word.doc2bow(text) for text in texts]
print('corpus has been created!')


pickle.dump(id2word, open(f'{data_dir}{sep}pickle{sep}id2word_ym_all.pickle', 'wb'))
pickle.dump(corpus, open(f'{data_dir}{sep}pickle{sep}corpus_ym_all.pickle', 'wb'))



