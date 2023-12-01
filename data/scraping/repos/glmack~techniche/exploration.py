#!/usr/bin/env python
# coding: utf-8

# ## Techniche - Topic Modelling

# In[14]:


import pandas as pd
import numpy as np

import gensim
import gensim.corpora as corpora
from gensim.corpora import mmcorpus
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.ldamodel import LdaModel
from gensim.models import AuthorTopicModel, atmodel
from gensim.test.utils import common_dictionary, datapath, temporary_file
from smart_open import smart_open

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, punkt, RegexpTokenizer, wordpunct_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer

import json
from pandas.io.json import json_normalize
import requests
import re
import os
import calendar
import sys

from test_model import tokenize_docs, clean_docs, lower_words, remove_stopwords, get_topics#, (TODO) Lee convert_bytes

from smart_open import smart_open

import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim

from pprint import pprint

import pyspark
import pyspark.sql.functions as F


# In[15]:


get_ipython().run_line_magic('load_ext', 'autoreload')

# pd.set_option('display.max_colwidth', -1)
pd.options.display.max_columns = 10
pd.set_option('display.max_rows', 10)


# In[33]:


np.random.seed(3)


# In[34]:


# uncomment to download stop words from nltk and language package from spacy
# nltk.download('stopwords')
# nltk.download('punkt')
# !python -m spacy download en


# ### Import Data

# #### Import data from PatentsView API

# In[35]:


# patents endpoint
endpoint_url = 'http://www.patentsview.org/api/patents/query'

# build list of possible fields that endpoint request will return
df = pd.read_excel("/Users/lee/Documents/techniche/techniche/data/patents_view_patents_fields.xlsx")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
pat_fields = df.api_field_name.values.tolist()


# #### Import initial dataset

# In[36]:


# build query - initial small dataset

query={"_or":[{"_text_phrase":{"patent_title":"natural language"}},
              {"_text_phrase":{"patent_abstract":"natural language"}}]}
# uncomment to use alternate query options
# query={"cpc_subgroup_id":"G06T3/4046"}
# query = {"_and":[{"_gte":{"patent_date":"2017-01-01"}},{"_lte":{"patent_date":"2017-01-31"}}]}
# query={"_and":
#         [{"_or":
#             [{"_text_phrase":{"patent_title":"machine learning"}}
#             ,{"_text_phrase":{"patent_abstract":"machine learning"}}]}
#         ,{"_and":
#       [{"patent_year":2016}]}]}
# query={"_or":[{"_text_phrase":{"patent_title":"natural language"}},{"_text_phrase":{"patent_abstract":"natural language"}}]}
# uncomment to use alternate query options
# query={"cpc_subgroup_id":"G06T3/4046"}
# query = {"_and":[{"_gte":{"patent_date":"2017-01-01"}},{"_lte":{"patent_date":"2017-01-31"}}]}
# query={"_and":
#         [{"_or":
#             [{"_text_phrase":{"patent_title":"natural language"}}
#             ,{"_text_phrase":{"patent_abstract":"natural language"}}]}
#         ,{"_and":
#       [{"patent_year":2016}]}]} 
# query = {"_and":[{"_gte":{"patent_date":"2017-01-01"}},{"_lte":{"patent_date":"2017-01-31"}}]}
fields=pat_fields
options={"per_page":100}
sort=[{"patent_date":"desc"}]

params={'q': json.dumps(query),
        'f': json.dumps(fields),
        'o': json.dumps(options),
        's': json.dumps(sort)}

# request and results
response = requests.get(endpoint_url, params=params)
status = response.status_code
print("status:", status)
results = response.json()
count = results.get("count")
total_pats = results.get("total_patent_count")
print("patents on current page:",count,';', "total patents:",total_pats)


# #### Structure data

# In[37]:


# extract metadata from response
print("status code:", response.status_code,';', "reason:", response.reason)
total_patent_count = results["total_patent_count"]
patents_per_page = results['count']
print("total_patent_count:",total_patent_count,';', "patents_per_page:", patents_per_page)

# extract data from response
data_resp = results['patents']
# data_resp[0]

raw_df = pd.DataFrame(data_resp)
raw_df.head(3)


# #### Subset dataframe

# In[38]:


# subset dataframe - comment/uncomment to include fields
df = raw_df[['patent_number', 
         'patent_date', 
         'patent_title',
         'patent_abstract', 
         'patent_firstnamed_assignee_id',
         'patent_firstnamed_assignee_location_id',
         'patent_firstnamed_assignee_latitude',
         'patent_firstnamed_assignee_longitude',
         'patent_firstnamed_assignee_city',
         'patent_firstnamed_assignee_state',
         'patent_firstnamed_assignee_country', 
         'patent_firstnamed_inventor_id',
         'patent_firstnamed_inventor_location_id',
         'patent_firstnamed_inventor_latitude',
         'patent_firstnamed_inventor_longitude',
         'patent_firstnamed_inventor_city',
         'patent_firstnamed_inventor_state',
         'patent_firstnamed_inventor_country',
         'patent_year', 
         'patent_type', 
         'patent_kind',
         'inventors'
            ]]
df.head(3)


# #### Explore data

# In[39]:


# 561 different assignees
len(df.patent_firstnamed_assignee_id.unique())


# #### Create new column

# In[40]:


# create new column that combines the patent title and the patent abstract columns into a single string
df['patent_title_abstract'] = df.patent_title + ' ' + df.patent_abstract
df.patent_title_abstract.head(3)


# In[41]:


df.sort_values(by=['patent_date'])


# In[42]:


text_data = df.patent_title_abstract.tolist()
text_data[:3]


# In[43]:


# partition data
len(text_data)
text_train = text_data[:round(len(text_data)*.8)]
text_test = text_data[round(len(text_data)*.8):]
print(len(text_data), len(text_train), len(text_test), len(text_train)+len(text_test))


# ### Pre-process text data

# In[44]:


# uncomment to download stop words from nltk and language package from spacy
# nltk.download('stopwords')
# nltk.download('punkt')
# !python -m spacy download en


# #### Tokenize

# In[45]:


# tokenize documents

def tokenize_docs(docs):
    tokenized_docs = []
    for doc in docs:
        tokenized_docs.append(word_tokenize(doc))
    return tokenized_docs

tokenized_docs = tokenize_docs(text_train)


# #### Clean punctuation

# In[46]:


# clean punctuation
def clean_docs(tokenized_docs):
    clean_docs = []
    for doc in tokenized_docs:
       clean_docs.append([word for word in doc if word.isalpha()])  
    return clean_docs


# In[47]:


cleaned_data = clean_docs(tokenized_docs)
cleaned_data[0]


# #### Convert to lowercase

# In[48]:


# convert to lowercase
def lower_words(docs):
    lowered_words = []
    for doc in docs:
        lowered_words.append([word.lower() for word in doc])
    return lowered_words

lowered_data = lower_words(cleaned_data)
lowered_data[0]


# #### Clean stopwords

# In[49]:


# clean stopwords

stop_words = stopwords.words('english')


# In[50]:


def filter_stopwords(docs):
    filtered_docs = []
    for doc in docs:
       filtered_docs.append([word for word in doc if word not in stop_words])
    return filtered_docs

# remove stopwords
filtered_data = filter_stopwords(lowered_data)
filtered_data
# TODO (Lee) - resolve un-lowered stopwords "A" and "An", 'By', 'The'


# #### Create dictionary and convert tokens into frequency counts by document

# In[51]:


# specify corpus - list of patent-list of tokenized words
texts = filtered_data


# In[52]:


filtered_data


# In[57]:


# build dictionary – a mapping between words and their integer ids
id_to_word = corpora.Dictionary(filtered_data)
# id_to_word[]


# In[58]:


# .dfs returns frequency of documents containing given token in tuple (token_id, count of documents that contain this token)
id_to_word.dfs.items()


# In[59]:


# apply term document frequency - convert docs in corpus to bag-of-words format, a list of (token_id, token_count) tuples
corpus_train = [id_to_word.doc2bow(text) for text in texts]
len(corpus_train)


# In[60]:


# inspect formatted term-doc frequency for word in text in corpus
[[(id_to_word[id], freq) for id, freq in text] for text in corpus_train]


# ### Model - model #1

# In[114]:


# TODO (Lee) - deprecation warnings
# construct LDA model
model_lda = LdaModel(corpus=corpus_train,
                     id2word=id_to_word,
                     num_topics=10, 
                     random_state=100,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)


# In[115]:


# # extracts topics for given document from Gensim
# def get_topics(doc, k=5, model_lda=model_lda):
#     topic_id = sorted(model_lda[doc][0], key=lambda x: -x[1])
#     top_k_topics = [x[0] for x in topic_id[:k]]
#     return [(i, model_lda.print_topic(i)) for i in top_k_topics]


# In[116]:


# `get_document_topics()` returns topic probability distribution for given document
topic_dist_675_a = model_lda.get_document_topics(corpus_train[15])
pprint(sorted(topic_dist_675_a))


# In[117]:


topicid = 3
model_lda.get_topic_terms(topicid, topn=10)


# In[118]:


text_train[doc_id]


# In[119]:


doc_id = 15
topic_dist_15_b = sorted(get_topics(corpus_train[doc_id], k=10)), text_train[doc_id]
pprint(topic_dist_15_b)


# In[107]:


# text = 'virtual dictionary lexicon enablement voice'.split()
text_input = 'smart assistant transformer model translation'.split()


# In[78]:


id_to_word.doc2bow(text_input)


# In[101]:


get_topics(id_to_word.doc2bow(text_input), k=10)


# In[80]:


def get_documents()


# In[81]:


text_train[0]


# In[82]:


model_lda[doc]


# In[83]:


model_lda.do_estep(chunk, state=None)


# In[84]:


# print keywords in n topics
sorted(model_lda.show_topics(), key=lambda x: x[1])


# In[85]:


# print keywords in n topics
sorted(model_lda.print_topics(), key=lambda x: x[1])


# In[86]:


# print keywords in n topics
sorted(model_lda.print_topics(), key=lambda x: x[1])


# In[87]:


# print keywords in n topics
sorted(model_lda.print_topics(), key=lambda x: x[0])


# In[88]:


# show_topic() returns n most important/relevant words, and their weights, that comprise given topic
pprint(model_lda.show_topic(1, topn=10))


# In[89]:


pprint(model_lda.show_topics(num_topics=5, num_words=10))


# ### Evaluate - model #1

# In[91]:


# calculate perplexity metrics
perplexity = model_lda.log_perplexity(corpus_train)
perplexity


# In[92]:


# TODO (Lee) - confirm that filtered_data is indeed the correct dataset to pass to texts param
# calculate coherence metric
coherence = CoherenceModel(model=model_lda, texts=filtered_data, dictionary=id_to_word, coherence='c_v')
coherence_1 = coherence.get_coherence()
coherence_1


# In[94]:


# calculate coherence metric or each of the n topicss
coherence_1 = coherence.get_coherence_per_topic()
coherence_1


# In[97]:


# explore topics
pyLDAvis.enable_notebook()
viz_topics_1 = pyLDAvis.gensim.prepare(model_lda, corpus_train, id_to_word)
viz_topics_1
# TODO (Lee) - salient vs relevant terms in pyLDA ?


# ### Model 2-  Mallet model

# In[120]:


# uncomment to download Mallet topic model
# !wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
# update this path
path_mallet = 'data/mallet-2.0.8/bin/mallet'


# In[121]:


model_2 = gensim.models.wrappers.LdaMallet(path_mallet, corpus=corpus_train, num_topics=25, id2word=id_to_word)


# In[123]:


# topics
pprint(model_2.show_topics(formatted=False))


# In[124]:


# calculate coherence metric
coherence_model_2 = CoherenceModel(model=model_2, texts=filtered_data, dictionary=id_to_word, coherence='c_v')
coherence_model_2 = coherence_model_2.get_coherence()
coherence_model_2


# ### Model 3 - Author topic model

# #### pre-process

# In[125]:


tokenized_docs_at = tokenize_docs(text_data)
cleaned_data_at = clean_docs(tokenized_docs_at)
lowered_data_at = lower_words(cleaned_data_at)
filtered_data_at = filter_stopwords(lowered_data_at)


# In[126]:


len(filtered_data_at)


# #### build dictionary and corpus from processed text

# In[127]:


# build dictionary
id_to_word_at = corpora.Dictionary(filtered_data_at)

# build corpus
texts_at = filtered_data_at

# apply term document frequency - converts docs in corpus to bag-of-words format via list of (token_id, token_count) tuples
corpus_at = [id_to_word_at.doc2bow(text) for text in texts_at]


# In[128]:


(next(iter(id_to_word_at.items())))


# In[129]:


type(id_to_word_at.keys()[0])


# In[130]:


type(id_to_word_at.values())


# #### construct inventor-doc mapping from nested inventors column in json api response

# In[131]:


# extract nested inventors table from api response
df_inventors = json_normalize(results['patents'], record_path=['inventors'], meta=['patent_number', 'patent_date'])
df_inventors = df_inventors[['inventor_id', 'patent_number', 'patent_date']]
df_inventors.sort_values(by=['patent_date'])
df_inventors.pop("patent_date")
df_inventors.head(3)


# In[132]:


# TODO (Lee) - resolve workaround
# df_idx = df
# df_idx['idx'] = df.index
# df_idx
# # df_idx_1 = df_idx[['patent_number', 'idx', 'inventors']]
# df_idx_2 = df_idx_1.set_index('patent_number')
# df_idx_2.pop('inventors')
# df_idx_2
# df_pat_idx = df_idx_2.T.to_dict('records')
# for i in df_pat_idx:
#     df_pat_idx = dict(i)
# df_pat_idx

# df_pat_idx = df_idx_2.T.to_dict('records')
# for i in df_pat_idx:
#     df_pat_idx = dict(i)
# df_pat_idx


# In[133]:


dict_pat2inv =df_inventors.set_index('patent_number').T.to_dict('list')
# dict_pat2inv


# In[134]:


# for k, v in pat2inv.items():
#     name_dict[new_key] = name_dict.pop(k)
#     time.sleep(4)

# pprint.pprint(name_dict)

# d = {'x':1, 'y':2, 'z':3}
# d1 = {'x':'a', 'y':'b', 'z':'c'}

# dict((d1[key], value) for (key, value) in d.items())
# {'a': 1, 'b': 2, 'c': 3}

# idx_pat_map = df.patent_number.to_dict()
# idx_pat_map = {str(key): value for key, value in idx_pat_map.items()}
# import itertools
# x = list(itertools.islice(idx_pat_map.items(), 0, 4))
# x[:4]


# In[135]:


pat2inv_dict = {k: list(v) for k,v in df_inventors.groupby("patent_number")["inventor_id"]}


# In[136]:


# {k: list(v) for k,v in df_pat2inv.groupby("patent_number")["inventor_id"]}


# In[137]:


# df2 = df_inventors.groupby("patent_number")["inventor_id"]


# In[138]:


# df3 = df_idx_pat_inv_map.groupby("patent_number")["inventor_id"]


# In[139]:


pat2inv = {k: list(str(v)) for k,v in df_inventors.groupby("patent_number")["inventor_id"]}
len(pat2inv.items())


# In[140]:


pat2inv.items()
type(next(iter(pat2inv)))


# In[141]:


pat2inv_2 = {str(k): list(v) for k,v in df_inventors.groupby("patent_number")["inventor_id"]}
len(pat2inv_2)


# In[142]:


patdf2inv_2 = dict((df_inventors[key], value) for (key, value) in pat2inv.items())
patdf2inv_2


# In[143]:


patdf2inv = dict((df_pat_idx[key], value) for (key, value) in pat2inv.items())
patdf2inv


# #### Construct author-topic model

# In[144]:


# construct author-topic model
model_at = AuthorTopicModel(corpus=corpus_at,
                         doc2author=patdf2inv,
                         id2word=id_to_word_at, 
                         num_topics=25)


# In[145]:


# construct vectors for authors
author_vecs = [model_at.get_author_topics(author) for author in model_at.id2author.values()]
author_vecs


# In[146]:


# inspect topic distribution for author with id# 7788103-1
# each topic has a probability of being expressed given the particular inventor, but only the ones above a certain threshold are shown.

model_at['7788103-1']


# In[147]:


# def show_author(name):
#     print('\n%s' % name)
#     print('Docs:', model.author2doc[name])
#     print('Topics:')
#     pprint([(topic_labels[topic[0]], topic[1]) for topic in model[name]])


# In[148]:


# calculate per-word bound, which is a measure of the model's predictive performance (reconstruction error?)

build doc2author dictionary

doc2author = atmodel.construct_doc2author(model.corpus, model.author2doc)


# In[149]:



doc2author = atmodel.construct_doc2author(model.corpus, model.author2doc)


# In[150]:


gensim.models.atmodel.construct_author2doc(doc2author)
# construct mapping from author IDs to document IDs

Parameters:	doc2author (dict of (int, list of str)) – Mapping of document id to authors.
Returns:	Mapping of authors to document ids.
Return type:	dict of (str, list of int)


# In[151]:


gensim.models.atmodel.construct_doc2author(corpus, author2doc)
construct mapping from document IDs to author IDs

Parameters:	
corpus (iterable of list of (int, float)) – Corpus in BoW format.
author2doc (dict of (str, list of int)) – Mapping of authors to documents.
Returns:	
Document to Author mapping.

Return type:	
dict of (int, list of str)


# ### Appendix

# #### Import full dataset from PatentsView API

# In[ ]:


# uncomment to use

# def get_patents_by_month(begin_date,end_date, pats_per_page):
#     """ requests patent data from PatentsView API by date range"""
#     endpoint_url = 'http://www.patentsview.org/api/patents/query'
#     page_counter=1
#     data = []
#     results = {}
#     count=1
    
#     for i in range(round(100000/pats_per_page)): # TODO (Lee) - replace with datetime for begin_date to end_date
        
#         if count ==0:
#             print("error/complete")
#             break
            
#         elif count > 0:     
#             # build query
#             query = {"_and":[{"_gte":{"patent_date":"2017-01-01"}},{"_lte":{"patent_date":"2017-01-31"}}]}
#             fields=pat_fields
#             options={"page": page_counter, "per_page":pats_per_page}
#             sort=[{"patent_date":"desc"}]
#             params={'q': json.dumps(query),
#                     'f': json.dumps(fields),
#                     'o': json.dumps(options),
#                     's': json.dumps(sort)
#                         }
    
#             # request and results
#             response = requests.get(endpoint_url, params=params)
#             status = response.status_code
#             print("status:", status,';',"page_counter:",page_counter, ";", "iteration:",i)
#             results = response.json()
#             count = results.get("count")
#             total_pats = results.get("total_patent_count")
#             print("patents on current page:",count,';', "total patents:",total_pats)
#             data.extend(results)
#             page_counter+=1
        
#     return data
#             # TODO (Lee) results =  json.loads(response.content)
#             # TODO (Lee) places.extend(results['results'])
#             # TODO (Lee) time.sleep(2)


# CPC fields for block 1 of query:
# Y10S-706 OR 
# G06N-003 OR 
# G06N-005/003:G06N-005/027 OR 
# G06N- 007/005:G06N-007/06 OR 
# G06N-099/005 OR
# G06T2207/20081 OR
# G06T2207/20084 OR
# G06T-003/4046 OR
# G06T-009/002 OR
# G06F-017/16 OR
# G05B-013/027 OR
# G05B- 013/0275 OR
# G05B-013/028 OR
# G05B-013/0285 OR
# G05B-013/029 OR
# G05B-013/0295 OR
# G05B-2219/33002 OR
# G05D-001/0088 OR
# G06K-009 OR
# G10L-015 OR
# G10L-017 OR
# G06F-017/27:G06F-017/2795 OR
# G06F-017/28:G06F-017/289 OR
# G06F-017/30029:G06F- 017/30035 OR
# G06F-017/30247:G06F-017/30262 OR 
# G06F-017/30401 OR
# G06F-017/3043 OR 
# G06F-017/30522:G06F-017/3053 OR 
# G06F-017/30654 OR 
# G06F-017/30663 OR
# G06F-017/30666 OR 
# G06F-017/30669 OR
# G06F-017/30672 OR 
# G06F-017/30684 OR
# G06F-017/30687 OR 
# G06F-017/3069 OR 
# G06F-017/30702 OR
# G06F-017/30705:G06F- 017/30713 OR
# G06F-017/30731:G06F-017/30737 OR
# G06F-017/30743:G06F-017/30746 OR 
# G06F-017/30784:G06F-017/30814 OR
# G06F-019/24 OR G06F-019/707 OR
# G01R- 031/2846:G01R-031/2848 OR
# G01N-2201/1296 OR
# G01N-029/4481 OR
# G01N-033/0034 ORG01R-031/3651ORG01S-007/417ORG06N-003/004:G06N-003/008 ORG06F- 011/1476 OR 
# G06F-011/2257 OR 
# G06F-011/2263 OR 
# G06F-015/18 OR
# G06F-2207/4824 OR
# G06K-007/1482 OR
# G06N-007/046 OR
# G11B-020/10518 OR
# G10H-2250/151 OR
# G10H-2250/311 OR
# G10K-2210/3024 OR
# H01J-2237/30427 OR
# H01M-008/04992 OR
# H02H-001/0092 OR
# H02P-021/0014 OR
# H02P-023/0018 OR
# H03H-2017/0208 OR
# H03H- 2222/04 OR
# H04L-2012/5686 OR
# H04L-2025/03464 OR
# H04L-2025/03554 OR
# H04L- 025/0254 OR
# H04L-025/03165 OR
# H04L-041/16 OR
# H04L-045/08 OR
# H04N- 021/4662:H04N-021/4666 OR
# H04Q-2213/054 
# OR H04Q-2213/13343 OR
# H04Q-2213/343 OR
# H04R-025/507 OR
# G08B-029/186 OR
# B60G-2600/1876 OR
# B60G-2600/1878 OR
# B60G-2600/1879 OR
# B64G-2001/247 OR
# E21B-2041/0028 OR
# B23K-031/006 OR
# B29C- 2945/76979 OR
# B29C-066/965 OR
# B25J-009/161 OR
# A61B-005/7264:A61B-005/7267 OR
# Y10S-128/924 OR
# Y10S-128/925 OR
# F02D-041/1405 OR
# F03D-007/046 OR
# F05B- 2270/707 OR
# F05B-2270/709 OR
# F16H-2061/0081 OR
# F16H-2061/0084 OR
# B60W-030/06 OR
# B60W-030/10:B60W-030/12 OR
# B60W-030/14:B60W-030/17 OR
# B62D-015/0285 OR
# G06T-2207/30248:G06T-2207/30268 OR
# G06T-2207/30236 OR G05D-001 OR
# A61B- 005/7267 OR
# F05D-2270/709 OR
# G06T-2207/20084 OR
# G10K-2210/3038 OR
# G10L-025/30 OR
# H04N-021/4666 OR
# A63F-013/67 OR
# G06F-017/2282

# #### Import data from bulk download

# In[ ]:


# uncomment to download TSV files containing detailed patent descriptions from PatentsView 
# !wget http://data.patentsview.org/detail-description-text/detail-desc-text-2016.tsv.zip # 2016 - 3.0 GB zipped
# !wget http://data.patentsview.org/detail-description-text/detail-desc-text-2017.tsv.zip # 2017 - 2.8 GB zipped
# !wget http://data.patentsview.org/detail-description-text/detail-desc-text-2018.tsv.zip # 2018 - 1.6 GB zipped
# !wget http://data.patentsview.org/detail-description-text/detail-desc-text-2019.tsv.zip # 2019 - 0.7 GB zipped

# !unzip files
# unzip detail-desc-text-2016.tsv.zip
# unzip detail-desc-text-2017.tsv.zip
# unzip detail-desc-text-2018.tsv.zip
# unzip detail-desc-text-2019.tsv.zip

# def convert_bytes(num, suffix='B'):
#     """ convert bytes int to int in aggregate units"""
#     for unit in ['','K','M','G','T','P','E','Z']:
#         if abs(num) < 1024.0:
#             return "%3.1f%s%s" % (num, unit, suffix)
#         num /= 1024.0
#     return "%.1f%s%s" % (num, 'Yi', suffix)

# path = "data/"
# with os.scandir(path) as it:
#     for entry in it:
#         if not entry.name.startswith('.') and entry.is_file():
#             print(entry.name)

# # inspect unzipped file sizes

# convert_bytes(os.path.getsize("data/detail-desc-text-2016.tsv"))
# convert_bytes(os.path.getsize("data/detail-desc-text-2017.tsv"))
# convert_bytes(os.path.getsize("data/detail-desc-text-2018.tsv"))
# convert_bytes(os.path.getsize("data/detail_desc_text_2019.tsv"))

#### Spark workflow

# create SparkSession/SparkContext as entry point to Dataset/DataFrame API

# spark = pyspark.sql.SparkSession.builder.getOrCreate()
# sc = spark.sparkContext
# sc

# doc by doc read of large tsv files

# with open "data/detail-desc-text-2018.tsv" as f_in:
#     with open('data/2018_shard.tsv', 'w') as f_out:
#         for line in f_in:
#             if contains_keywords(line):
#                 f_out.write(line)

# from functools import reduce

# df = reduce(lambda x,y: x.unionAll(y), 
#             [spark.read.format('csv')
#                        .load(f, header="true", inferSchema="true") 
#              for f in files])
# df.show()

# files = ["data/detail-desc-text-2016.tsv", "data/detail-desc-text-2017.tsv", 
#          "data/detail-desc-text-2018.tsv", "data/detail-desc-text-2019.tsv"]

# uncomment to use

# df_2018 = (spark.read
#                .format("csv")
#                .option("delimiter", "\t")
#                .option('inferSchema', "true")
#                .load("data/detail-desc-text-2018.tsv")
#                .write
#                .format("parquet")
#                .save("df_2018.parquet"))

# df_2018 = (spark.read
#                .format("csv")
#                .option("delimiter", "\t")
#                .option('inferSchema', "true")
#                .load("data/detail-desc-text-2018.tsv"))

# dfp_2018 = pd.read_csv("data/detail-desc-text-2018.tsv", sep='\t', header=None)

# dfp_2018.columns = ['patent_number', 'desc_detail', 'len_detail']

# dfp_2018_nl.head(3)

# dfp_2018_nl = dfp_2018[dfp_2018['desc_detail'].str.contains('NLP')]

# df_2018.printSchema()

# schema = StructType([
#             StructField("_c0", IntegerType(), True),
#             StructField("_c1", StringType(), True),
#             StructField("_c2", IntegerType(), True)])

# df_2018 = spark.read.load('data/df_2018.parquet')

# # 160,249 rows in 2018 dataset
# df_2018.count()

# df_2018.rdd.getNumPartitions()

# # partition / batching ?
# df_2018.filter(df_2018._c1.contains("natural language")).count()

# query file directly with SQL

# query = """
# SELECT * FROM parquet.`data/df_2018.parquet` WHERE _c1 LIKE 'natural language' LIMIT 100
# """

# df_2018_nl = spark.sql(query)

# df_2018_nl.head(3)

# df_2018.columns

# df_2018.explain()

# df_2018.describe().show()

# df_2018.dtypes

# df_171819 = df_2017.union(df_2018).union(df_2019)

# df_171819.count()

# df_171819.head(3)

# df_2018.head(3)

# counts = df_2018.agg(F.countDistinct('_c0'))
# counts

# reviews_df.createOrReplaceTempView('reviews')

# output = spark.sql(query)

# show(output, n=1000)

# results = spark.sql(
#   "SELECT * FROM people")
# names = results.map(lambda p: p.name)

# df.rdd.isEmpty()

# df = (spark.read
#             .load("data/*.parquet")
#             .write
#             .format("parquet")
#             .save("df.parquet"))

# df_2019 = (spark.read
#                .format("csv")
#                .option("delimiter", "\t")
#                .option('inferSchema', "true")
#                .load("data/detail-desc-text-2018.tsv")
#                .write
#                .format("parquet")
#                .save("df_2019.parquet"))

# df_2019.head(3)

# type(df_2018)

# df_2016.head(3)

# df_2016 = (spark.read.format("csv")
#                .option("delimiter", "\t")
#                .option("header", "true")
#                .option('inferSchema', "true")
#                .load("data/detail-desc-text-2016.tsv")
#                .write
#                .format("parquet")
#                .save("data/df_2016.parquet"))

# df_2016.head(3)

# df = (spark.read.format("csv")
#            .option("delimiter", ",")
#            .infer
#            .load("data/df.csv"))

# df_2017 = (spark.read.format("csv")
#                .option("delimiter", "\t")
#                .option("header", "true")
#                .option('inferSchema', "true")
#                .load("data/detail-desc-text-2017.tsv")
#                .write
#                .format("parquet")
#                .save("df_2017.parquet"))

# df_2018 = spark.read.parquet("data/df_2018.parquet")

# df_2018.head(2)

# df_2018.persist()

# df_2018.take(2)

# df_2018.toPandas()


# #### Construct bigrams and trigrams

# In[62]:


# train bigram phrases model
bigram_model = Phrases(filtered_data, min_count=1, threshold=1)

# train trigram phrases model
trigram_model = Phrases(bigram_model[filtered_data], threshold=100)  

# bigrams
def bigrams(docs):
    """create bigrams"""
    return [bigram_model[doc] for doc in docs]

# initialize bigram and trigram models
bigram_model = gensim.models.phrases.Phraser(bigram_model)
trigram_model = gensim.models.phrases.Phraser(trigram_model)

bigrams(filtered_data)[0]

def trigrams(docs):
    """create trigrams"""
    return [trigram_model[bigram_model[doc]] for doc in docs]

trigrams(filtered_data)[0]


# #### Stem and Lemmatize

# In[63]:


def lemmatize_docs(docs, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """lemmatize documents"""
    lemmatized_docs = []
    for doc in docs: 
        lemmatized_docs.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return lemmatized_docs


# #### Alternate worklow to Create corpus and dictionary

# In[ ]:


# using spacy pipeline components
# build dictionary
id_to_word = corpora.Dictionary(processed_docs)

# build corpus
texts = processed_docs

# apply term document frequency
# converts documents in corpus to bag-of-words format, a list of (token_id, token_count) tuples
corpus = [id_to_word.doc2bow(doc) for doc in processed_docs]

 # build dictionary
id_to_word = corpora.Dictionary(filtered_data)

# build corpus
texts = filtered_data

# apply term document frequency
# converts documents in corpus to bag-of-words format, a list of (token_id, token_count) tuples
corpus = [id_to_word.doc2bow(text) for text in texts]

# view formatted corpus (term-doc-frequency)
[[(id_to_word[id], freq) for id, freq in text] for text in corpus][:1]


# #### Alternate workflow to pre-process text data

# In[ ]:


# # uncomment to download stop words from nltk and language package from spacy
# # nltk.download('stopwords')
# # nltk.download('punkt')
# # !python -m spacy download en

# # construct pipeline using Spacy Language object and associated pipeline/components
# nlp = spacy.load("en")
# pprint(nlp.pipeline)

# processed_docs = []   

# # process patent documents in pipeline
# for doc in nlp.pipe(text_train, n_threads=4, batch_size=100):
   
#     ents = doc.ents  # Named entities.

#     # Keep only words (no numbers, no punctuation).
#     # Lemmatize tokens, remove punctuation and remove stopwords.
#     doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

#     # Remove common words from a stopword list.
#     doc = [token for token in doc if token not in stop_words]

#     # Add named entities, but only if they are a compound of more than word.
#     doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
#     processed_docs.append(doc)

# processed_docs[0][:5]

# [token.text for token in doc]

# labels = set([w.label_ for w in doc.ents]) 

# for label in labels: 
#     entities = [cleanup(e.string, lower=False) for e in document.ents if label==e.label_] 
#     entities = list(set(entities)) 
#     print(label,entities)

# pre_processed_docs = []
# for doc in nlp.pipe(docs, n_threads=4, batch_size=100):
#     # Process document using Spacy NLP pipeline.
    
#     ents = doc.ents  # Named entities.

#     # Keep only words (no numbers, no punctuation).
#     # Lemmatize tokens, remove punctuation and remove stopwords.
#     doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

#     # Remove common words from a stopword list.
#     #doc = [token for token in doc if token not in STOPWORDS]

#     # Add named entities, but only if they are a compound of more than word.
#     doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
#     pre_processed_docs.append(doc)


# #### Alternate workflow to create corpus and dictionary

# In[65]:


# #uncomment to use

# # using spacy pipeline components
# # build dictionary
# id_to_word = corpora.Dictionary(processed_docs)

# # build corpus
# texts = processed_docs

# # apply term document frequency
# # converts documents in corpus to bag-of-words format, a list of (token_id, token_count) tuples
# corpus = [id_to_word.doc2bow(doc) for doc in processed_docs]

#  # build dictionary
# id_to_word = corpora.Dictionary(filtered_data)

# # build corpus
# texts = filtered_data

# # apply term document frequency
# # converts documents in corpus to bag-of-words format, a list of (token_id, token_count) tuples
# corpus = [id_to_word.doc2bow(text) for text in texts]

# # view formatted corpus (term-doc-frequency)
# [[(id_to_word[id], freq) for id, freq in text] for text in corpus][:1]


# #### Alternate worfklow for Model - model #1

# In[ ]:


# TODO (Lee) - deprecation warnings
# construct LDA model
model_lda = LdaModel(corpus=corpus,
                     id2word=id_to_word,
                     num_topics=25, 
                     random_state=100,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)

# print keywords in n topics
pprint(model_lda.print_topics())

# print top 10 keywords that comprise topic with index of 0
pprint(model_lda.print_topic(24))
# the most import keywords, and the respective weight, that form topic 0 are

# print top 10 keywords that comprise topic with index of 1
pprint(model_lda.print_topic(1))

# TODO (Lee) - infer topic from keywords?


# #### Alternate workflow to Evaluate - model #1

# In[ ]:


#uncomment to use

# calculate perplexity metrics
perplexity = model_lda.log_perplexity(corpus)
perplexity

# TODO (Lee) - confirm that filtered_data is indeed the correct dataset to pass to texts param
# calculate coherence metric
coherence = CoherenceModel(model=model_lda, texts=processed_docs, dictionary=id_to_word, coherence='c_v')
coherence_1 = coherence.get_coherence()
coherence_1

# TODO (Lee) - confirm that filtered_data is indeed the correct dataset to pass to texts param
# calculate coherence metric
coherence = CoherenceModel(model=model_lda, texts=filtered_docs, dictionary=id_to_word, coherence='c_v')
coherence_1 = coherence.get_coherence()
coherence_1

# calculate coherence metric or each of the n topicss
coherence_1 = coherence.get_coherence_per_topic()
coherence_1


# #### Worfklow re: coherence metric

# In[ ]:


# TODO (Lee)
# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     """
#     Compute c_v coherence for various number of topics

#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics

#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.wrappers.LdaMallet(path_mallet, corpus=corpus, num_topics=num_topics, id2word=id2word)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values

# model_list, coherence_values = compute_coherence_values(dictionary=id_to_word, corpus=corpus, texts=data, start=2, limit=40, step=6)


# #### Alternate workflow for Model 3 - Author topic model

# In[ ]:


# #uncomment to use

# # construct inventor-to-doc mapping as df from nested inventors column in json api response
# df_inventors = json_normalize(results['patents'], record_path=['inventors'], meta=['patent_number', 'patent_date'])
# df_inventors = df_inventors[['inventor_id', 'patent_number', 'patent_date']]
# df_inventors.sort_values(by=['patent_date'])
# df_inventors.head(3)

# df.head(3)

# # TODO (Lee) - resolve workaround
# df_idx = df
# df_idx['idx'] = df.index
# df_idx
# df_idx_1 = df_idx[['patent_number', 'idx', 'inventors']]
# df_idx_2 = df_idx_1.set_index('patent_number')
# df_idx_2.pop('inventors')
# df_idx_2
# df_pat_idx = df_idx_2.T.to_dict('records')
# for i in df_pat_idx:
#     df_pat_idx = dict(i)
# df_pat_idx

# df_pat_idx = df_idx_2.T.to_dict('records')
# for i in df_pat_idx:
#     df_pat_idx = dict(i)
# df_pat_idx

# df_inv_test = json_normalize(results['patents'], record_path=['inventors'], meta=['patent_number', 'patent_date'])
# df_inv_test.head(3)

# df_idx_pat_inv_map = df[['patent_number', 'inventors']]
# df_idx_pat_inv_map.head(3)

# # TODO (Lee) - find out how to get list of patents_view_field names from API - I did it accidentally but need to replicate

# df.patent_title_abstract[0]

# df[:3]

# df_inventors.set_index('inventor_id').T.to_dict('list')

# # for k, v in pat2inv.items():
# #     name_dict[new_key] = name_dict.pop(k)
# #     time.sleep(4)

# # pprint.pprint(name_dict)

# # d = {'x':1, 'y':2, 'z':3}
# # d1 = {'x':'a', 'y':'b', 'z':'c'}

# # dict((d1[key], value) for (key, value) in d.items())
# # {'a': 1, 'b': 2, 'c': 3}

# patdf2inv = dict((df_pat_idx[key], value) for (key, value) in pat2inv.items())
# patdf2inv

# pat2inv = {k: list(v) for k,v in df_inventors.groupby("patent_number")["inventor_id"]}
# pat2inv

# idx_pat_map = df.patent_number.to_dict()
# idx_pat_map = {str(key): value for key, value in idx_pat_map.items()}
# idx_pat_map

# #### Construct author-topic model

# # construct author-topic model
# model_at = AuthorTopicModel(corpus=corpus,
#                          doc2author=patdf2inv,
#                          id2word=id_to_word, 
#                          num_topics=25)

# # construct vectors for authors
# author_vecs = [model_at.get_author_topics(author) for author in model_at.id2author.values()]
# author_vecs

# # retrieve the topic distribution for an author using use model[name] syntax
# # each topic has a probability of being expressed given the particular author, but only the ones above a certain threshold are shown.

# model_at['7788103-1']

# # def show_author(name):
# #     print('\n%s' % name)
# #     print('Docs:', model.author2doc[name])
# #     print('Topics:')
# #     pprint([(topic_labels[topic[0]], topic[1]) for topic in model[name]])

# # calculate per-word bound, which is a measure of the model's predictive performance (reconstruction error?)

# build doc2author dictionary

# doc2author = atmodel.construct_doc2author(model.corpus, model.author2doc)

# from gensim.models import atmodel
# doc2author = atmodel.construct_doc2author(model.corpus, model.author2doc)

# gensim.models.atmodel.construct_author2doc(doc2author)
# # construct mapping from author IDs to document IDs.

# Parameters:	doc2author (dict of (int, list of str)) – Mapping of document id to authors.
# Returns:	Mapping of authors to document ids.
# Return type:	dict of (str, list of int)

# gensim.models.atmodel.construct_doc2author(corpus, author2doc)
# construct mapping from document IDs to author IDs

# Parameters:	
# corpus (iterable of list of (int, float)) – Corpus in BoW format.
# author2doc (dict of (str, list of int)) – Mapping of authors to documents.
# Returns:	
# Document to Author mapping.

# Return type:	
# dict of (int, list of str)


# #### Appendix - Model #1 - Evaluate - holdout set

# In[1]:


# appendix
# TODO (Lee) - evaluate on 1k documents **not** used in LDA training
doc_stream = (tokens for _, tokens in iter_wiki('./data/simplewiki-20140623-pages-articles.xml.bz2'))  # generator
test_docs = list(itertools.islice(doc_stream, 8000, 9000))


# #### Appendix - Model #1 - Evaluate - Doc split

# In[ ]:


# TODO (Lee) - split each document into two parts, and check that 1) topics of the first half are similar to 
topics of the second 2) halves of different documents are mostly dissimilar:


# In[ ]:


# TODO (Lee)
def intra_inter(model, test_docs, num_pairs=10000):
    # split each test document into two halves and compute topics for each half
    part1 = [model[id2word.doc2bow(tokens[: len(tokens) / 2])] for tokens in test_docs]
    part2 = [model[id2word.doc2bow(tokens[len(tokens) / 2 :])] for tokens in test_docs]
    
    # print computed similarities (uses cossim)
    print("average cosine similarity between corresponding parts (higher is better):")
    print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))

    random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
    print("average cosine similarity between 10,000 random parts (lower is better):")    
    print(np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs]))


# In[ ]:


# TODO (Lee)
print("LDA results:")
intra_inter(lda_model, test_docs)


# In[ ]:


#### Appendix - Model #1 - Evaluate - Log likelihood


# In[ ]:


#### Appendix - Model #1 - Evaluate - Alternate unimplemented workflow


# In[ ]:


# TODO (Lee)
# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     """
#     Compute c_v coherence for various number of topics

#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics

#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.wrappers.LdaMallet(path_mallet, corpus=corpus, num_topics=num_topics, id2word=id2word)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values

# model_list, coherence_values = compute_coherence_values(dictionary=id_to_word, corpus=corpus, texts=data, start=2, limit=40, step=6)


# In[ ]:


#### Appendix - Model 1 - Inference - Alternate workflows


# In[ ]:


# `get_document_topics()` returns topic probability distribution for given document
# topic_dist_675_a = model_lda.get_document_topics(corpus[50])
# pprint(sorted(topic_dist_50_a))


# In[ ]:


# topicid = 3
# model_lda.get_topic_terms(topicid, topn=10)


# In[ ]:


# text_train[doc_id]
# doc_id = 675
# topic_dist_675_b = sorted(get_topics(corpus[doc_id], k=10)), text_train[doc_id]
# pprint(topic_dist_675_b)


# In[ ]:


# From Gensim example - Alternate predict workflow - Create a new corpus, made of previously unseen documents.
# other_texts = [
#      ['computer', 'time', 'graph'],
#      ['survey', 'response', 'eps'],
#      ['human', 'system', 'computer']
#  ]
# other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
# unseen_doc = other_corpus[0]
# vector = lda[unseen_doc]  # get topic probability distribution for a document
# Update the model by incrementally training on the new corpus


# In[ ]:


# print keywords in n topics
sorted(model_lda.show_topics(), key=lambda x: x[1])

# print keywords in n topics
sorted(model_lda.print_topics(), key=lambda x: x[1])

# print keywords in n topics
sorted(model_l.print_topics(), key=lambda x: x[1])

# print keywords in n topics
sorted(model_1.print_topics(), key=lambda x: x[0])

# show_topic() returns n most important/relevant words, and their weights, that comprise given topic
pprint(model_1.show_topic(1, topn=10))

pprint(model_1.show_topics(num_topics=5, num_words=10))


# In[ ]:


#### Appendix - Model #2 - Evaluate - Perplexity
No implementation of log_perplexity method for LDAMallet


# In[ ]:


#### Appendix - Train Model #5


# In[ ]:


## Appendix - this was alternative partial workflow to create inv:doc mapping in model #5
# uncomment to construct inventor-to-doc mapping as df from nested inventors column in json api response
df_inventors = json_normalize(raw_data_1000, record_path=['inventors'], meta=['patent_number', 'patent_date'])
df_inventors = df_inventors[['inventor_id', 'patent_number', 'patent_date']]
df_inventors.sort_values(by=['patent_date'])
df_inventors.head(3)

