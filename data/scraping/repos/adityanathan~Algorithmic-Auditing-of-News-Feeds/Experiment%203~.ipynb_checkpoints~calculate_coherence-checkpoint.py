#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import pickle
import re
import timeit
import spacy
import copy

import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, HdpModel, LdaModel, LdaMulticore
from nltk.corpus import stopwords
import helper as he

input1 = '../../data/preprocessed_data/doc_indexes/demon.pkl'
input2 = '../../data/preprocessed_data/corpus_dict/demon_corp.pkl'
output = '../../data/temp/demon_confusion.pkl'

with open(input1,'rb') as f:
    texts,INITIAL_DOC_SIZE, DOC_TEMPORAL_INCREMENT = pickle.load(f)

with open(input2, 'rb') as f:
    data_lemmatized, _, _ = pickle.load(f)


# In[20]:


# for i in DOC_TEMPORAL_INCREMENT[:167]:
#     INITIAL_DOC_SIZE+=i
# DOC_TEMPORAL_INCREMENT=DOC_TEMPORAL_INCREMENT[-10:]
INITIAL_DOC_SIZE = DOC_TEMPORAL_INCREMENT[140]
DOC_TEMPORAL_INCREMENT = DOC_TEMPORAL_INCREMENT[141:]


# In[21]:


# Set Data State to that of existing model in simulation
data = data_lemmatized[:INITIAL_DOC_SIZE]
id2word = Dictionary(documents=data_lemmatized)
corpus = [id2word.doc2bow(doc) for doc in data]

# Building for the first time - To be considered as the starting/existing model in simulation.
lda = LdaMulticore(corpus, num_topics=35, id2word=id2word,
                   workers=3, chunksize=2000, passes=10, batch=False)

#Baseline Model
corpus_baseline = copy.deepcopy(corpus)
lda_baseline = copy.deepcopy(lda)


# In[ ]:


# The loop simulates arrival of new documents in batches where batch_size is defined in DOC_TEMPORAL_INCREMENT
doc_size = []
positive_arr = []

f2 = open(output, 'wb')

count = 0
doc_size_counter = INITIAL_DOC_SIZE
print('Total Corpus Length:',len(data_lemmatized))
for i in DOC_TEMPORAL_INCREMENT:
    # new_docs is the list of STEP_SIZE new documents which have arrived
    new_docs = data_lemmatized[doc_size_counter:doc_size_counter+i]
    doc_size_counter += i

    prev_corpus = copy.deepcopy(corpus)

    # Converting Documents to doc2bow format so that they can be fed to models
    corpus = [id2word.doc2bow(doc) for doc in new_docs]
    count += 1

    print('MODEL NO:'+str(count))
    lda.update(corpus)
    print('MODEL DONE')

    prev_corpus.extend(corpus)
    corpus = copy.deepcopy(prev_corpus)

    doc_size.append(i)
    positive_arr.append(he.calc_confusion_matrix(
        lda_baseline, lda, corpus))

    pickle.dump((positive_arr, doc_size), f2)

f2.close()


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(positive_arr)
plt.show()


# In[18]:


positive_arr


# In[ ]:




