# %%
# Imports
import json
import pickle
import collections
import random
import time
import os
from pprint import pprint

from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

dirpath = os.path.dirname(os.path.realpath('__file__'))
with open('./words_dictionary.json') as filehandle:
    words_dictionary = json.load(filehandle)
english_words = words_dictionary.keys()
stop_words = stopwords.words('english')
stop_words.extend(['https', 'http'])
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# %%
# Define functions

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
# %%
# Read in the file and then prepare docs for lda modeling


with open('./data/dataset_tweets_markets_followers.data', 'rb') as filehandle:
    dataset_tweets_markets_followers = pickle.load(filehandle)

labels = []
docs = []
for user in dataset_tweets_markets_followers:
    label = dataset_tweets_markets_followers[user]['label']
    doc = dataset_tweets_markets_followers[user]['tweets']

    labels += [label]
    docs += [doc]

id2word = corpora.Dictionary(docs)
# Idea: Keep only those tokens that appear in at least 10% of the documents
id2word.filter_extremes(no_below=int(0.1*len(docs)))
corpus = [id2word.doc2bow(doc) for doc in docs]

# %%
# Define LDA model and save output to file

lda_model_random_users = compute_lda(corpus, id2word)
lda_model_random_users.save('./ldamodels/market_followers/model.model')
pprint(lda_model_random_users.print_topics())

# Also save the corpus because gensim does not save this automatically
with open('./ldamodels/market_followers/corpus.corpus', 'wb') as filehandle:
    pickle.dump(corpus, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# Generate feature vectors given a document and an lda model

feature_vectors = list(lda_model_random_users.get_document_topics(corpus))
# feature_vectors = [doc_topics for doc_topics, word_topics, word_phis in
#                                                                  topics]
augmented_feature_vectors = get_augmented_feature_vectors(feature_vectors)

# %%
# Define the binary classifier (SVC with RBF kernel)

X_train, X_test, y_train, y_test = train_test_split(augmented_feature_vectors,
                                                    labels,
                                                    test_size=0.33,
                                                    random_state=100)
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Accuracy score in distinguishing between human and commercial:',
      accuracy_score(y_test, y_pred))

# Save the classifier
with open('./models/commercial_follower_classifier.clf', 'wb') as filehandle:
    pickle.dump(clf, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
