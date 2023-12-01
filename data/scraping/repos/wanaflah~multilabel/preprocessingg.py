# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:48:19 2019

@author: wanmaflah.mzubi
"""

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import nltk
from nltk.stem import WordNetLemmatizer
en_stop = set(nltk.corpus.stopwords.words('english'))
import re


def preprocess(data):
    print("Starting preprocessing")
    print("Step 0 : Importing libraries used.")
        #to tokenize
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(re.sub('@[^\s]+','',sentence)), deacc=True))  # deacc=True removes punctuations
    
    data_words = list(sent_to_words(data))
    
    print("Step 1 : Tokenized done.")
    
    ## Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=3, threshold=50) # higher threshold fewer phrases.
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in en_stop] for doc in texts]  
    
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    
    # Remove Stop Words
    data_words = remove_stopwords(data_words)
    
    print("Step 2 : Remove stop words done.")
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words)
    
    print("Step 3 : Bigram analysis done.")

    preprocessed_data = data_words_bigrams
    
    print("Preprocessing done.")

    return preprocessed_data