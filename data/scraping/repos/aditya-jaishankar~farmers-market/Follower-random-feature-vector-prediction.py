# %% [markdown]
# # This file is used for Calculating the feature vector matrices for the set 
# of followers and the set of random users

# %% 
# Imports

import json
import glob
import pickle
import collections
import random
from tqdm import tqdm as tqdm
import time
import os
dirpath = os.path.dirname(os.path.realpath('__file__'))
from pprint import pprint

# import logging
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
# logging.root.level = logging.INFO

# NLP imports
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['https', 'http'])
import re
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# To check later if a words is in english or not. Note that to include some
# additional words as stop words, I just removed them from this dictionary
with open('./words_dictionary.json') as filehandle:
    words_dictionary = json.load(filehandle)
english_words = words_dictionary.keys()

# Visualization imports

# Other imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

# %%
# Defining functions


def get_augmented_feature_vectors(feature_vectors):
    """
    Takes in the feature vector list of list and augments it. gensim does not
    actually put a 0 for topics that have 0 probability so I need to manually
    add it in to build my feature vector. 
    input: accepts the feature vectors output by gensim. It is a list of 
    tuples - one list entry per document and tuple are (topic, probability)
    pairs.
    returns: Augmented feature vectors as list of list. Each list entry 
    corresponds to one document, with the i-th element in the inner list
    corresponding to the probability that the document was generated with 
    topic i.
    """
    augmented = []
    for i, vector in enumerate(feature_vectors): # each vector is a list of tuples
        topics = [tup[0] for tup in vector]
        for t in range(10):
            if t not in topics:
                feature_vectors[i].append((t, 0))
        new_feature_vector = sorted(feature_vectors[i], key=lambda tup: tup[0])
        augmented.append([tup[1] for tup in new_feature_vector])
    return augmented


def get_docs(d, market):
    """
    Accepts a market and then returns the documents for the market. A document
    is a list of of word lists for each user in the market city i.e. it is a list of lists.
    Each outer list is a follower and the innner list is the cleaner, tokenized, depunkt, 
    lematized set of words for that follower.
    """
    docs = []
    for user in d[market]:
        text_list = d[market][user]['fulltext']
        docs.append(text_list)
    return docs
# %%
# Loading all the necessary data

# Load the NYC lda pretrained model
market_index = 1  # We will use the model for NYC
filename_model = './ldamodels/market' + str(market_index) + '/model.model'
lda_model = gensim.models.ldamodel.LdaModel.load(filename_model)
# load the corpus
filename_corpus = './ldamodels/market' + str(market_index) + '/corpus.corpus'

with open(filename_corpus, 'rb') as filehandle:
    corpus = pickle.load(filehandle)

# Random users dict
with open('./data/lda_dict_random_users.data', 'rb') as filehandle:
    lda_dict_random_users = pickle.load(filehandle)

# Followers master dict
with open('./data/master_dict.data', 'rb') as filehandle:
    master_dict = pickle.load(filehandle)


# %%
# Choose 30 random users

labels = []
docs_random = []
for user in list(lda_dict_random_users.keys())[:30]:  # Just select 30 random users
    try:
        label = lda_dict_random_users[user]['label']
        doc = lda_dict_random_users[user]['fulltext']

        labels += [label]
        docs_random += [doc]
    except:
        pass

# %%
# Chose 30 followers

list_markets = pd.read_excel('./list_of_farmers_markets.xlsx')
list_markets = list_markets.sort_values(by=['Num_Followers'], ascending=False)
list_markets = list_markets.reset_index(drop=True)

# Based on some data inspection, we remove portland, GrowNYC and Madison. 
list_markets = list_markets.drop([1, 4, 9]).reset_index(drop=True)
markets = list(master_dict.keys())

docs_followers = get_docs(master_dict, markets[market_index])[:30]  # Choose only 30 docs

# %%
# Calculate the augmented feature matrix for the original followers

old_id2word = lda_model.id2word

followers_corpus = [old_id2word.doc2bow(doc) for doc in docs_followers]  
topics_followers = lda_model.get_document_topics(followers_corpus, 
                                                 per_word_topics=True)
feature_vectors_followers = [doc_topics for doc_topics, word_topics, word_phis 
                             in topics_followers]
augmented_feature_vectors_followers = get_augmented_feature_vectors(
                                                     feature_vectors_followers)

# %%
# Calculate the augmented feature matrix for the random people

random_corpus = [old_id2word.doc2bow(doc) for doc in docs_random]  
topics_random = lda_model.get_document_topics(random_corpus, 
                                                 per_word_topics=True)
feature_vectors_random = [doc_topics for doc_topics, word_topics, word_phis 
                             in topics_random]
augmented_feature_vectors_random = get_augmented_feature_vectors(
                                                        feature_vectors_random)

pprint('follower feature vector')
pprint(augmented_feature_vectors_followers)

pprint('random feature vector')
pprint(augmented_feature_vectors_random)

# %%
# t1 = time.time()
# lda_model_random_users = compute_lda(corpus, id2word)
# t2 = time.time()
# print('Time elapsed:', t2-t1)
# lda_model_random_users.save('./ldamodels/random_users/model.model')
# pprint(lda_model_random_users.print_topics())




# %% [markdown]

## Generating feature vectors given a document and given the lda model

# %%
topics = lda_model_random_users.get_document_topics(corpus, 
                                                    per_word_topics=True)
feature_vectors = [doc_topics for doc_topics, word_topics, word_phis in topics]

# Note that get_document_topics only returns cases where the probability is
# non zero, so I will have to manually go in and add zeros.
augmented_feature_vectors = get_augmented_feature_vectors(feature_vectors)

# %% [markdown]

# At this point, I have a list of feature vectors and a list of labels that
# correspond to that feature vector, which can now be fed into a binary
# classifier. I am going to start with a random forest and then see if I can or
# should do better than that. 

# %% [markdown]

## Binary Classification Implementation

# I have my feature vectors (list of list) in `augmented_feature_vectors` and 
# the corresponding labels in `labels`. This is ready to fed into a binary
# classifier. First, I will split my data into a test_train_split.

# %%

X_train, X_test, y_train, y_test = train_test_split(augmented_feature_vectors,
                                                    labels,
                                                    test_size=0.33,
                                                    random_state=100)
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Accuracy score in distinguishing between human and commercial:', 
        accuracy_score(y_test, y_pred))

# %% 

# save the binary classifier

with open('./models/commercial-filter-classifier.model', 'wb') as filehandle:
    pickle.dump(clf, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# At this point I have a trained binary classifier that is presumably able to
# detect other farmers markets, farms, and other commercial users. We next use
# this trained model to further tune our input into the LDA when looking at 
# predictions by city.