# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:47:52 2019

@author: yanqi
"""
import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 4 - Capstone\\AmazonReview'
os.chdir(proj_path)
import pandas as pd
pd.set_option('display.max_colwidth', -1)  # to view entire text in any column
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
import string
import gzip
from pprint import pprint

import gensim
from gensim import corpora
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim   # for visualizing found topics
from model_utils import qc_dict, out_topics_docs, check_topic_doc_prob, topn_docs_by_topic


os.environ['MALLET_HOME'] = "C:/Users/yanqi/Library/mallet-2.0.8"
mallet_path = "C:/Users/yanqi/Library/mallet-2.0.8/bin/mallet"

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)  # suppressing deprecation warnings when running gensim LDA

select_cat = 'Electronics->Accessories & Supplies->Audio & Video Accessories->TV Accessories & Parts->TV Ceiling & Wall Mounts'.split('->')

# useful attributes & methods: dictionary.token2id to get mapping, dictionary.num_docs 
df = pd.read_csv(select_cat[-1] + "_processed.csv", index_col= 0)
df.reset_index(drop=True, inplace = True)
reviews = df['review_lemmatized'].copy()
reviews = reviews.apply(lambda x: x.split())

# Dictionary expects a list of list (of tokens)
dictionary = corpora.Dictionary(reviews)
dictionary.filter_extremes(no_below=3)  # remove terms that appear in < 3 documents, memory use estimate: 8 bytes * num_terms * num_topics * 3

# number of terms
nd = dictionary.num_docs
nt = len(dictionary.keys())
print("number of documents", nd)
print("number of terms", nt)

qc_dict(dictionary)

# create document term matrix (corpus), it's a list of nd elements, nd = the number of documents
# each element of DTM (AKA corpus) is a list of tuples (int, int) representing (word_index, frequency)
DTM = [dictionary.doc2bow(doc) for doc in reviews]

%time ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=DTM, num_topics=10, id2word=dictionary)

# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=reviews, dictionary=dictionary, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)

len(ldamallet[DTM])  # mallet produces a dense doc-topic probability matrix, this could be problematic for memory
topics_docs_dict = out_topics_docs(ldamallet, DTM)

# check doc_topic probability distribution
for t in sorted(topics_docs_dict.keys()):
    test_prob = check_topic_doc_prob(topics_docs_dict, t)
    print(test_prob.describe(),"\n")
    
# examine each topic by topic key words, number of generated documents, document probabilities, docs with top probabilities
topic_num = 0
print(ldamallet.show_topic(topicid=topic_num))
print("topic", topic_num, "has", len(topics_docs_dict[topic_num]),"documents")
print("Distribution of probabilities of documents being generated from this topic:")
doc_prob = check_topic_doc_prob(topics_docs_dict, topic_num)
print(doc_prob.describe(),"\n")
top_docprobs = topn_docs_by_topic(topics_docs_dict,topic_num, 10)
idxs = pd.Series([x[0] for x in top_docprobs])
probs = pd.Series([x[1] for x in top_docprobs])
texts = pd.Series([df['review_no_html'][i] for i in idxs])
products = pd.Series([df['title'][i] for i in idxs])
asins = pd.Series([df['asin'][i] for i in idxs])
top_docs_df = pd.concat([asins, products, idxs, probs, texts], axis = 1)
top_docs_df.columns = ['asin','product','doc_id', 'prob_from_topic','reviewText']
pd.set_option('display.max_columns', 500)
top_docs_df[['asin','product','doc_id', 'prob_from_topic']]
top_docs_df['reviewText']

# topics with only unigrams
# topic 0: pulled out a subcategory antennas, and bracket form of TV wall mount
# *topic 1: quality of the intructions, how easy is it to follow for installation
# topic 3: many mention sound bars / bracket, positioning of the mount, not super helpful
# topic 4: related to installation and weight bearing, studs, bolts    
# topic 6: keywords not coherent, reviewText emphasize simple mount that works for small size TVs
# topic 7: pulled out a subcategory mount arms (product names has arm in it)
# *topic 8: both product titles and reviews focus on weight bearing capacity of the mount
# *topic 9: overall very positive reviews, satisfied with price, value, quality, easy install



