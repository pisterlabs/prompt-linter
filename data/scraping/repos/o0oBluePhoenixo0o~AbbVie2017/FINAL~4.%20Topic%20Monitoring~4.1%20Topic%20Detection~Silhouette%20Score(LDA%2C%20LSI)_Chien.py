##### Before running this script, please put this file into the data repository so as to run it.

#### Determine number of topics K for topic model: LDA & LSI
### Silhouette score is built by "sklearn" package & K-means
'''
Given a set of documents (twitter tweets), this script
* Tokenize the preprocessed text
* Apply Latent Dirichlet Allocation(LDA) & Latent Semantic Indexing(LSI)
* Normalize the document-term Matrix
* Apply K-means to segment the documents
Reference can be seen: https://github.com/alexperrier/datatalks/blob/master/twitter/twitter_lsa.py
'''

import sys
import re
import os
import json
import csv
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import gensim
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pyLDAvis.gensim as gensimvis
import pyLDAvis

### Load balanced Twitter dataset
df_postn = pd.read_csv('10000_twitter_preprocessing.csv', encoding = 'UTF-8', sep = ',', index_col = 0)
df_postn = df_postn.sort_values(['created_time'])
df_postn.index = range(len(df_postn))
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def tokenize(doc):
    reurl = re.sub(r"http\S+", "", str(doc))
    tokens = ' '.join(re.findall(r"[\w']+", reurl)).lower().split()
    x = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    x = ' '.join(x)
    stop_free = " ".join([i for i in x.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word, pos = 'n') for word in punc_free.split())
    normalized = " ".join(lemma.lemmatize(word, pos = 'v') for word in normalized.split())
    word = " ".join(word for word in normalized.split() if len(word) > 3)
    postag = nltk.pos_tag(word.split())
    poslist = ['NN', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']
    wordlist = ['co', 'https', 'http', 'rt', 'www', 've', 'dont', "i'm", "it's",'kf4pdwe64k','co', 'https', 'http', 'rt', 'www', 've', 'dont', "i'm", "it's",'kf4pdwe64k','nhttps','cant','didnt']
    adjandn = [word for word, pos in postag if pos in poslist and word not in wordlist and len(word) > 3]
    return ' '.join(adjandn)

import datetime
import dateutil.relativedelta
def dateselect(day):
    d = datetime.datetime.strptime(str(datetime.date.today()), "%Y-%m-%d")
    d2 = d - dateutil.relativedelta.relativedelta(days=day)
    df_time = df_postn['created_time']
    df_time = pd.to_datetime(df_time)
    mask = (df_time > d2) & (df_time <= d)
    period = df_postn.loc[mask]
    return period

### Load doc_term_matrix, corpus
corpus = list(df_postn['re_message'])
directory = "doc_clean.txt"
if os.path.exists(directory):
    with open("doc_clean.txt", "rb") as fp:  # Unpickling
        doc_clean = pickle.load(fp)
else:
    doc_clean = [tokenize(doc).split() for doc in corpus]
    with open("doc_clean.txt", "wb") as fp:  # Pickling
        pickle.dump(doc_clean, fp)
directory = "corpus.dict"
if os.path.exists(directory):
    dictionary = corpora.Dictionary.load('corpus.dict')
else:
    dictionary = corpora.Dictionary(doc_clean)
    dictionary.save('corpus.dict')
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
tfidf = models.TfidfModel(doc_term_matrix)
finalcorpus = tfidf[doc_term_matrix]

### LDA Silhouette Score
## Number of Topics K: range from 20 to 70
import gensim
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from gensim.matutils import corpus2dense
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from time import time
import numpy as np
import operator
cl = ['blue', 'red', 'green', '#89f442', 'cyan', 'magenta', 'yellow', 'black']
fig = plt.figure(figsize=(12,9), dpi=800)
r = 0
for i in [20,30,40,50,60,70]:
    ldamodel = LdaModel(finalcorpus, num_topics = i, id2word = dictionary, update_every = 10, 
                        chunksize = 2000, passes = 10, alpha = 0.05)
    corpus_lda = ldamodel[finalcorpus]
    corpus_lda_dense = corpus2dense(corpus_lda, i)
    k = np.array(corpus_lda_dense).transpose()
    nlzr = Normalizer(copy=False)
    kk = nlzr.fit_transform(k)
    silhouette = []
    for j in range(5,80,3):
        km = KMeans(n_clusters = j, init = 'k-means++', max_iter = 100, n_init = 4, verbose = False, random_state = 10)
        label = km.fit(kk)
        label_lda = label.labels_
        lda_silhouette_avg = silhouette_score(kk, label_lda, sample_size = 1000)  
        silhouette.append(lda_silhouette_avg)
    plt.plot(list(range(5, 80,3)), silhouette, color = cl[r], linewidth = 1, linestyle = "-", label = str(i))
    plt.xticks(list(range(5, 80,5)))
    plt.xlabel('#K-means Cluster', fontsize = 18)
    plt.ylabel('Silhouette Score', fontsize = 16)
    plt.legend(loc = 'upper right', frameon = False)
    r += 1
    plt.savefig("lda_silhouette_tw_2070.png", dpi = 100)
plt.show()

## Number of Topics K: range from 20 to 40
## Narrow the range of K
cl = ['blue', 'red', 'green', '#89f442', 'cyan', 'magenta', 'yellow', 'black']
fig = plt.figure(figsize = (12,9), dpi = 800)
r = 0
for i in [20,25,30,35,40]:
    ldamodel = LdaModel(finalcorpus, num_topics = i, id2word = dictionary, update_every = 10,
                        chunksize = 2000, passes = 10, eta = None, alpha = 0.05)
    corpus_lda = ldamodel[finalcorpus]
    corpus_lda_dense = corpus2dense(corpus_lda, i)
    k = np.array(corpus_lda_dense).transpose()
    nlzr = Normalizer(copy = False)
    kk = nlzr.fit_transform(k)
    silhouette = []
    for j in range(5,50,3):
        km = KMeans(n_clusters = j, init = 'k-means++', max_iter = 100, n_init = 4, verbose = False, random_state = 10)
        label = km.fit(kk)
        label_lda = label.labels_
        lda_silhouette_avg = silhouette_score(kk, label_lda, sample_size = 1000)
        silhouette.append(lda_silhouette_avg)
    plt.plot(list(range(5, 50,3)), silhouette, color = cl[r], linewidth = 1, linestyle = "-", label = str(i))
    plt.xticks(list(range(5, 50,5)))
    plt.xlabel('#K-means Cluster', fontsize = 18)
    plt.ylabel('Silhouette Score', fontsize = 16)
    plt.legend(loc = 'upper right', frameon = False)
    r += 1
    plt.savefig("lda_silhouette_tw_2040.png", dpi = 100)
plt.show()

### LSI Silhouette Score
## Number of Topics K: range from 20 to 70
cl = ['blue', 'red', 'green', '#89f442', 'cyan', 'magenta', 'yellow', 'black']
fig = plt.figure(figsize = (12,9), dpi = 800)
r = 0
for i in [20,30,40,50,60,70]:
    lsimodel = LsiModel(finalcorpus, id2word = dictionary, num_topics = i, chunksize = 2000)
    corpus_lsi = lsimodel[finalcorpus]
    corpus_lsi_dense = corpus2dense(corpus_lsi, i)
    k = np.array(corpus_lsi_dense).transpose()
    nlzr = Normalizer(copy = False)
    kk = nlzr.fit_transform(k)
    silhouette = []
    for j in range(5,80,3):
        km = KMeans(n_clusters = j, init = 'k-means++', max_iter = 100, n_init = 4, verbose = False, random_state = 10)
        label = km.fit(kk)
        label_lsi = label.labels_
        lsi_silhouette_avg = silhouette_score(kk, label_lsi, sample_size = 1000)  
        silhouette.append(lsi_silhouette_avg)
    plt.plot(list(range(5, 80,3)), silhouette, color = cl[r], linewidth = 1, linestyle = "-", label = str(i))
    plt.xticks(list(range(5, 80,5)))
    plt.xlabel('#K-means Cluster', fontsize = 18)
    plt.ylabel('Silhouette Score', fontsize = 16)
    plt.legend(loc = 'upper right', frameon = False)
    r += 1
    plt.savefig("lsi_silhouette_tw_2070.png", dpi = 100)
plt.show()

## Number of Topics K: range from 20 to 40
## Narrow the range of K
cl = ['blue', 'red', 'green', '#89f442', 'cyan', 'magenta', 'yellow', 'black']
fig = plt.figure(figsize = (12,9), dpi = 800)
r = 0
for i in [20,25,30,35,40]:
    lsimodel = LsiModel(finalcorpus, id2word = dictionary, num_topics = i, chunksize = 2000)
    corpus_lsi = lsimodel[finalcorpus]
    corpus_lsi_dense = corpus2dense(corpus_lsi, i)
    k = np.array(corpus_lsi_dense).transpose()
    nlzr = Normalizer(copy = False)
    kk = nlzr.fit_transform(k)
    silhouette = []
    for j in range(5, 50, 3):
        km = KMeans(n_clusters = j, init = 'k-means++', max_iter = 100, n_init = 4, verbose = False, random_state = 10)
        label = km.fit(kk)
        label_lsi = label.labels_
        lsi_silhouette_avg = silhouette_score(kk, label_lsi, sample_size = 1000) 
        silhouette.append(lsi_silhouette_avg)
    plt.plot(list(range(5, 50, 3)), silhouette, color = cl[r], linewidth = 1, linestyle = "-", label = str(i))
    plt.xticks(list(range(5, 50, 5)))
    plt.xlabel('#K-means Cluster', fontsize = 18)
    plt.ylabel('Silhouette Score', fontsize = 16)
    plt.legend(loc = 'upper right', frameon = False)
    r += 1
    plt.savefig("lsi_silhouette_tw_2040.png", dpi = 100)
plt.show()
