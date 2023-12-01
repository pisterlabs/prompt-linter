#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:25:54 2017

@author: abazaga
"""

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

import glob
from ggplot import *
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import MySQLdb
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from nltk.cluster.kmeans import *
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk import *
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import text
from itertools import cycle, islice
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.random_projection import sparse_random_matrix
import nltk.stem
import re
from matplotlib import cm
import matplotlib.colors
import warnings
import gensim
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.utils import lemmatize
from pprint import pprint
from nltk.corpus import stopwords
import pyLDAvis.gensim


import _pickle as cPickle

pattern = re.compile("^([A-Z][0-9]+)+$")

stemmer = EnglishStemmer()
analyzernew = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzernew(doc))

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" | ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
def process_texts(texts):
    """
    Function to process texts. Following are the steps we take:
    
    1. Stopword Removal.
    2. Collocation detection.
    3. Lemmatization (not stem since stemming can reduce the interpretability).
    
    Parameters:
    ----------
    texts: Tokenized texts.
    
    Returns:
    -------
    texts: Pre-processed tokenized texts.
    """
    texts = [[word for word in line if word not in stops] for line in texts]
    texts = [bigram[line] for line in texts]
    texts = [[word.split(b'/')[0] for word in lemmatize(' '.join(line), min_length=5)] for line in texts]
    return texts
    
    
num_clusters = 20
max_iterations = 3000
# calculate tf-idf of texts
s = set(stopwords.words('english'))
#my_stop_words = text.ENGLISH_STOP_WORDS.union(s).union(["bioinformatics","science","berlin","verlag","heidelberg","ms","copyright","ieee","society","biomed","springer","elsevier","factor","change","data","central","relationship","term","nucleic","acid","research","right","conclusions","validation","site","step","size","selection","vector","behalf","target","work","context","type","domain","power","licensee","license","identification","background","conclusion","role","difference","problem","model","occurrence","compraison","similarity","level","search","state","body","connectivity","distribution","random","graph","requirement","control","normalization","normal","chosen","output","regression","class","time","space","rate","performance","procedure","process","interface","scale","matrix","characteristic","accession","need","determination","error","date","addition","completeness","risk","probe","help","line","image","segmentation","measure","transformation","execution","result","results","query","web","investigation","investigate","evidence","profile","interaction","knowledge","activity","browser","scheme","corpus","ontology","support","lack","project","user","feedback","technology","template","assignment","server","platform","manner","management","storage","cross","classification","classifier","prediction","predictor","funcionality","failure","study","ii","iii","i","sum","access","progress","information","run","batch","literature","www","html","xml","research","language","log","feature","ratio","quality","spot","order","test","http","org","com","range","text","significance","oxford","set","accuracy","use","paper","method","published","rights","press","university","new","based","number","large","developed","reserved","motivation","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","author","authors","used","using","approach","summary","results"])
#my_stop_words = text.ENGLISH_STOP_WORDS.union(s).union(["event","technique","application","graphic","graphics","day","method","methods","acm","format","mean","source","scientist","issue","subject","area","java","researcher","year","challenge","resource","case","advance","solution","taylor","francis","kegg","patient","gpu","parameter","value","bioinformatics","science","berlin","verlag","heidelberg","ms","copyright","ieee","society","biomed","springer","elsevier","factor","change","data","central","relationship","term","nucleic","acid","research","right","conclusions","validation","site","step","size","selection","vector","behalf","target","work","context","type","domain","power","licensee","license","identification","background","conclusion","role","difference","problem","model","occurrence","compraison","similarity","level","search","state","body","connectivity","distribution","random","graph","requirement","control","normalization","normal","chosen","output","regression","class","time","space","rate","performance","procedure","process","interface","scale","matrix","characteristic","accession","need","determination","error","date","addition","completeness","risk","probe","help","line","image","segmentation","measure","transformation","execution","result","results","query","web","investigation","investigate","evidence","profile","interaction","knowledge","activity","browser","scheme","corpus","ontology","support","lack","project","user","feedback","technology","template","assignment","server","platform","manner","management","storage","cross","classification","classifier","prediction","predictor","funcionality","failure","study","ii","iii","i","sum","access","progress","information","run","batch","literature","www","html","xml","research","language","log","feature","ratio","quality","spot","order","test","http","org","com","range","text","significance","oxford","set","accuracy","use","paper","method","published","rights","press","university","new","based","number","large","developed","reserved","motivation","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","author","authors","used","using","approach","summary","results"])

# Open database connection
db = MySQLdb.connect("-","-","-","-" )

cursor = db.cursor()

cursor.execute("SELECT abstract,title from - WHERE type IN ('Article','Conference Paper') ORDER BY RAND()")
# Fetch a single row using fetchone() method.

texts_list = []
title_list = []
for row in cursor:
    texts_list.append(str(row[0]))
    title_list.append(str(row[1]))

wordnet_lemmatizer = WordNetLemmatizer()

# Load them from file
#title_list = cPickle.load(open("title_list2.pickle", "rb"))
print(len(title_list))
tf_idf_vectorizer = cPickle.load(open("tfidfvectorizer2.pickle", "rb"))
tf_idf_matrix = cPickle.load(open("tfidfmatrix2.pickle", "rb"))
X_dense = tf_idf_matrix.todense()
#texts_list = cPickle.load(open("texts_list2.pickle", "rb"))
print(len(texts_list))
#
print("TDF Vectorizer, matrix and texts loaded from file")

######

bigram = gensim.models.Phrases(texts_list)  # for bigram collocation detection
stops = set(stopwords.words('english'))  # nltk stopwords list

texts_list = process_texts(texts_list)
dictionary = Dictionary(texts_list)
corpus = [dictionary.doc2bow(text) for text in texts_list if text !=[] and text != [[]]]
print(len(corpus))

### LSI

lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)
print(lsimodel.show_topics(num_topics=5))  # Showing only the top 5 topics
lsitopics = lsimodel.show_topics(formatted=False)

### HDP
hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
print(hdpmodel.show_topics())
hdptopics = hdpmodel.show_topics(formatted=False)

### LDA
ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

