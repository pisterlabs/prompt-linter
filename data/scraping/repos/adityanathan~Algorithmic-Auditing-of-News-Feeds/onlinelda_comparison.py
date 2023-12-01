#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# No of Documents = 17849 <br>
# Start Date = 2016-10-31 <br>
# End Date = 2017-12-31 <br>
# <br>
# Google Alerts Start-Date = Feb-26-2019 <br>
#
# * 2016-12-31 - 9442
# * 2017-02-28 - 12710
# * 2017-04-30 - 14339
# * 2017-06-30 - 15445
# * 2017-08-31 - 16287
# * 2017-10-31 - 16998
# * 2017-12-31 - 17849

# In[1]:


# imports

from __future__ import print_function
from time import time
import numpy as np
import pandas as pd
import pickle
import random

# Sklearn
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import re, nltk, spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
from pyLDAvis import gensim as pyLDAvisgensim
import matplotlib.pyplot as plt


# Loading the entire corpus containing 17849 documents

# In[2]:

f = open("output_online.txt","w+")
doc_set = []
with open("demon_combined.pkl", "rb") as pickle_in:
    while True:
        try:
            doc_set.append(pickle.load(pickle_in))
        except:
            break


# Choose the two models for which you would like to build the confusion matrix.<br>
# Input: No of Documents in Corpus, Best topic no.
#
# * 09442 - best topic no = 34 (start = 10, end = 40)
# * 12710 - best topic no = 37
# * 14339 - best topic no = 31
# * 15445 - best topic no = 31
# * 16287 - best topic no = 29
# * 16998 - best topic no = 25
# * 17849 - best topic no = 37

# In[3]:

    answers_positive = []
    answers_negative = []
    model_nos=(0,6)


    # In[4]:


    settings=[(9442,34),(12710,37),(14339,31),(15445,31),(16287,29),(16998,25),(17849,37)]
    offset = 10
    corpus_length_1 = settings[model_nos[0]][0]
    corpus_length_2 = settings[model_nos[1]][0]
    best_1 = settings[model_nos[0]][1] - offset
    best_2 = settings[model_nos[1]][1] - offset
    print(corpus_length_1)
    print(corpus_length_2)
    print(best_1+offset)
    print(best_2+offset)


    # Loading the corresponding Models and Corpus.

    # In[5]:


    doc_set_1 = doc_set[:corpus_length_1]
    doc_set_2 = doc_set[:corpus_length_2]
    doc_set_diff = doc_set[corpus_length_1:corpus_length_2+1]

    model_1_batch=[]
    model_2_batch=[]
    with open("Models/ldamodels_demon_"+str(corpus_length_1)+".pkl", "rb") as pickle_in:
        while True:
            try:
                model_1_batch.append(pickle.load(pickle_in))
            except:
                break
    with open("Models/ldamodels_demon_"+str(corpus_length_2)+".pkl", "rb") as pickle_in:
        while True:
            try:
                model_2_batch.append(pickle.load(pickle_in))
            except:
                break
    # In[6]:


    # NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts,bigram_mod):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts,trigram_mod):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def preprocess(doc_set):
        df = pd.DataFrame(doc_set)
        df.columns = ['text']

        # Convert to list
        data = df.text.values.tolist()

        # Remove @ characters, newlines, single quotes
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
        data = [re.sub('\s+', ' ', sent) for sent in data]
        data = [re.sub("\'", "", sent) for sent in data]
        pprint(data[:1])
        data_words = list(sent_to_words(data))
        print(data_words[:1])

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram is to use the Phraser interface
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # See bigrams and trigrams example
        print(trigram_mod[bigram_mod[data_words[0]]])

        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        print(data_lemmatized[:1])

        return data_lemmatized


# In[2]:



    data_lemmatized_1=preprocess(doc_set_1)
    print(data_lemmatized_1[1])


    # In[8]:


    # Create Dictionary
    id2word_1 = corpora.Dictionary(data_lemmatized_1)

    #Filter words
    id2word_1.filter_extremes(no_below=5, no_above=0.95, keep_n=1800, keep_tokens=None)

    # Create Corpus
    texts = data_lemmatized_1

    # Term Document Frequency
    corpus_1 = [id2word_1.doc2bow(text) for text in texts]

    # View
    print(corpus_1[:1])


    # In[9]:


# In[3]:


data_lemmatized_2=preprocess(doc_set_2)


# In[10]:


# Create Dictionary
id2word_2 = corpora.Dictionary(data_lemmatized_2)

#Filter words
id2word_2.filter_extremes(no_below=5, no_above=0.95, keep_n=1800, keep_tokens=None)

# Create Corpus
texts = data_lemmatized_2

# Term Document Frequency
corpus_2 = [id2word_2.doc2bow(text) for text in texts]

# View
print(corpus_2[:1])


# In[4]:


data_lemmatized_diff=preprocess(doc_set_diff)

id2word_diff = corpora.Dictionary(data_lemmatized_diff)

id2word_diff.filter_extremes(no_below=5, no_above=0.95, keep_n=1800, keep_tokens=None)

texts = data_lemmatized_diff

corpus_diff = [id2word_diff.doc2bow(text) for text in texts]

print(corpus_diff[:1])


# In[6]:


coherencemodel = CoherenceModel(model=model_1_batch[0][0][best_1], texts=data_lemmatized_1, dictionary=id2word_1, coherence='c_v')
print(coherencemodel.get_coherence())


# In[17]:


model_1_batch[0][0][best_1]


# In[7]:


model_2_online = model_1_batch[0][0][best_1]
model_2_online.update(corpus_diff)
print(model_2_online)


# In[8]:


coherencemodel2 = CoherenceModel(model=model_2_online, texts=data_lemmatized_2, dictionary=id2word_2, coherence='c_v')
print(coherencemodel2.get_coherence())


# In[9]:


lda_corpus_1 = [max(prob,key=lambda y:y[1])
                    for prob in model_2_online[corpus_2] ]
lda_corpus_2 = [max(prob,key=lambda y:y[1])
                    for prob in model_2_batch[corpus_2] ]
print(lda_corpus_1==lda_corpus_2)

print(len(lda_corpus_1))
print(len(lda_corpus_2))
max_corpus_size = max(len(lda_corpus_1),len(lda_corpus_2))
print(max_corpus_size)

total_permutations=max_corpus_size*(max_corpus_size-1)/2 #nC2 combinations


# In[ ]:


model_2==model_1


# In[ ]:


positive=0
negative=0
for i in range(max_corpus_size):
    for j in range(i+1,max_corpus_size):
        if(lda_corpus_1[i][0]==lda_corpus_1[j][0] and lda_corpus_2[i][0]==lda_corpus_2[j][0]):
            positive = positive+1
        elif(lda_corpus_1[i][0]!=lda_corpus_1[j][0] and lda_corpus_2[i][0]==lda_corpus_2[j][0]):
            negative = negative+1
        elif(lda_corpus_1[i][0]==lda_corpus_1[j][0] and lda_corpus_2[i][0]!=lda_corpus_2[j][0]):
            negative = negative+1
        elif(lda_corpus_1[i][0]!=lda_corpus_1[j][0] and lda_corpus_2[i][0]!=lda_corpus_2[j][0]):
            positive=positive+1

f.write(str((positive+negative==total_permutations)))

f.write("% of positives = "+str(round(positive*100/total_permutations,2))+"%")
f.write("% of negatives = "+str(round(negative*100/total_permutations,2))+"%")

answers_positive.append(round(positive*100/total_permutations,2))
answers_negative.append(round(negative*100/total_permutations,2))

f.write(str(answers_positive))
f.write(str(answers_negative))


# In[ ]:


print(answers_positive)
print(answers_negative)


#     Online LDA (Before (Batch unupdated with online) vs After(Batch Updated with Online)) - same model from batch
# 
# | Models Tested (AuB) | Positive% | Negative% |
# | --- | --- | --- |
# | (0,1) | 99.63 | 0.37 |
# | (1,2) | 99.41 | 0.59 |
# | (2,3) | 99.73 | 0.27 |
# | (3,4) | 99.66 | 0.34 |
# | (4,5) | 99.67 | 0.33 |
# | (5,6) | 99.87 | 0.13 |
# | (0,6) | 99.54 | 0.4 |
# 
#     Online LDA (Batch i, Batch i+1) Compared after Batch i trained with corpus_dif with online LDA

# In[ ]:




