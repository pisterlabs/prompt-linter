# %% [markdown]
# # Training a binary classifier to identify accounts that are likely 
# commercial business vs those that are likely real human users
# 
# The goal of this file is to be able to identify which of the followers that I
#  selected are commercial followers or otherwise small businesses. This 
# corrupts my input user base with non-people, so this is an attempt to remove
# these followers.
# %% [markdown]
# ## Imports

# %%
# General imports
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
import pyLDAvis
import pyLDAvis.gensim
# pyLDAvis.enable_notebook()
import matplotlib.pyplot as plt

# Other imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

# %% [markdown]

## All functions

# %%
def compute_lda(corpus, id2word, k=10, alpha='auto'):
    """
    Performs the LDA and returns the computer model.
    Input: Corpus, dictionary and hyperparameters to optimize
    Output: the fitted/computed LDA model
    """
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                                id2word=id2word,
                                                num_topics=k,
                                                random_state=100,
                                                # update_every=1,
                                                chunksize=5,
                                                passes=100,
                                                alpha=.01,
                                                iterations=100,
                                                per_word_topics=True)
    return lda_model

def visualize_LDA(model, corpus):
    """
    This function accepts an lda model and a corpus of words and uses pyLDAvis
    to prepare a visualization and then save to html. 
    input: an lda model and a corpus of words
    returns: None
    """
    LDAvis_prepared = pyLDAvis.gensim.prepare(model, corpus,
                                              dictionary=model.id2word,
                                              mds='tsne')
    vis_filename = './LDAvis_prepared/random_users/LDAvis.html'
    pyLDAvis.save_html(LDAvis_prepared, vis_filename)
    pyLDAvis.show(LDAvis_prepared)
    return None


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

# %% [markdown]

## Loading the data

# %%

with open('./data/lda_dict_random_users.data', 'rb') as filehandle:
    lda_dict_random_users = pickle.load(filehandle)

# %% [markdown]

## Generating the docs, corpuses and labels

# Note that there are about 320 documents with label 1 (yes to farmers market) 
# and about 480 documents with label 0 (not farmers market).

# %%

# it seems that some of my entries don't have label, probably because it was
# skipped in a previous try, except clause. So I will try and catch this 
# up front:
labels = []
docs = []
for user in lda_dict_random_users:
    try:
        label = lda_dict_random_users[user]['label']
        doc = lda_dict_random_users[user]['fulltext']

        labels += [label]
        docs += [doc]
    except:
        pass

id2word = corpora.Dictionary(docs)
# Idea: Keep only those tokens that appear in at least 10% of the documents
id2word.filter_extremes(no_below=int(0.1*len(docs)))
corpus = [id2word.doc2bow(doc) for doc in docs]

# %%
t1 = time.time()
lda_model_random_users = compute_lda(corpus, id2word)
t2 = time.time()
print('Time elapsed:', t2-t1)
lda_model_random_users.save('./ldamodels/random_users/model.model')
pprint(lda_model_random_users.print_topics())




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