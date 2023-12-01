# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:55:59 2020

@author: Alina Arseniev
"""
from __future__ import division
import cython
import gensim
import math
from gensim.models import coherencemodel
import pickle
from scipy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from seaborn import boxplot 
from nltk.tokenize import word_tokenize
from itertools import combinations
import matplotlib.pyplot as plt
#import pandas_profiling
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
#from spellchecker import SpellChecker
from gensim.models import Word2Vec
import pandas as pd
import re
import string, re

from gensim import corpora, models, similarities #calc all similarities at once, from http://radimrehurek.com/gensim/tut3.html
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, KeyedVectors
from random import seed, sample
from ksvd import ApproximateKSVD #pip or conda install ksvd



def reconst_qual(w2vmodel, dictionary_mat, gamma_mat):
    #reconstruct the word vectors
    reconstructed = gamma_mat.dot(dictionary_mat) #reconstruct word vectors and add back in mean(?). but note that reconstructed norm is still around 0-1, not 1, is that an issue?
    #e1 = norm(w2vmodel.wv.vectors - reconstructed) #total reconstruction error, larger means MORE error. norm as specified here takes frobenius norm of error matrix.


    #total VARIANCE in the data: sum of squares 
    squares3= w2vmodel.wv.vectors-np.mean(w2vmodel.wv.vectors, axis=1).reshape(-1,1) #https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
    #sst3= np.sum([i.dot(i) for i in squares3] ) #same as below

    sst3= np.sum(np.square(squares3))


    #total sum of squared ERRORS/residuals
    e3= [reconstructed[i]-w2vmodel.wv.vectors[i] for i in range(0,len(w2vmodel.wv.vectors))]  #https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
    #sse3= np.sum([i.dot(i) for i in e3] ) #same as below
    sse3= np.sum(np.square(e3))

    #R^2: 1- (SSE / SST )
    r2= 1- (sse3 /  sst3) #https://stats.stackexchange.com/questions/184603/in-pca-what-is-the-connection-between-explained-variance-and-squared-error


    #compute root mean square error
    rmse=  math.sqrt(np.mean(np.square(e3)))



    return(sse3, rmse, r2) #https://stats.stackexchange.com/questions/184603/in-pca-what-is-the-connection-between-explained-variance-and-squared-error

def topic_diversity(w2vmodel, dictionary_mat, top_n=25):

    topwords=[] #list of list, each innter list includes top N words in that topic

    for i in range(0, len(dictionary_mat)): #set to number of total topics
        topwords.extend([i[0] for i in w2vmodel.wv.similar_by_vector(dictionary_mat[i],topn=top_n)]) #set for top N words 
        #print(w2vmodel.wv.similar_by_vector(dictionary[i],topn=N))

    uniquewords= set(topwords)
    diversity = len(uniquewords)/len(topwords)
    return(diversity)

def coherence_centroid(w2vmodel, dictionary_mat, top_n): #eventually, combine with topic diversity into a class, since redundancy. this is based on: Aletras, Nikolaos, and Mark Stevenson. "Evaluating topic coherence using distributional semantics." Proceedings of the 10th International Conference on Computational Semantics (IWCS 2013)–Long Papers. 2013.
    minsim= []
    meansim=[]
    for k in dictionary_mat: #set to number of total topics
        words= [i[0] for i in w2vmodel.wv.similar_by_vector(k,topn=25)]
        topwordsvecs= [w2vmodel.wv[i] for i in words] #vecs for top closest words
        medvec= np.mean(topwordsvecs, axis=0) #using median rather than mean, since then less swayed by outliers
        sims= [abs(cosine_similarity(i.reshape(1,-1), medvec.reshape(1,-1))[0]) for i in topwordsvecs]
        minsim.append(np.min(sims))
        meansim.append(np.mean(sims))
    return(np.mean(meansim)) #average min, and average mean


def coherence_pairwise(w2vmodel, dictionary_mat, top_n): #eventually, combine with topic diversity into a class, since redundancy. this is based on: Aletras, Nikolaos, and Mark Stevenson. "Evaluating topic coherence using distributional semantics." Proceedings of the 10th International Conference on Computational Semantics (IWCS 2013)–Long Papers. 2013.
    #minsim= []
    meansim=[] #list of coherence of atoms in the model
    for k in dictionary_mat: #set to number of total topics
        words= [i[0] for i in w2vmodel.wv.similar_by_vector(k,topn=25)]
        topwordsvecs= [w2vmodel.wv[i] for i in words] #vecs for top closest words
        combo_sims= [abs(cosine_similarity(l[0].reshape(1,-1), l[1].reshape(1,-1))[0])[0] for l in combinations(topwordsvecs, 2)]
        #medvec= np.mean(topwordsvecs, axis=0) #using median rather than mean, since then less swayed by outliers
        #sims= [abs(cosine_similarity(i.reshape(1,-1), medvec.reshape(1,-1))[0]) for i in combos]
        #minsim.append(np.min(sims))
        meansim.append(np.mean(combo_sims)) #append coherence of this atom
    return(np.mean(meansim)) #mean of coherences of atoms in the model

