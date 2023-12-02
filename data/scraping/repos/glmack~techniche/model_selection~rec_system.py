#!/usr/bin/env python
# coding: utf-8

# ## Recommendation System
# Collaborative filtering with implicit feedback based on latent factors. Prepare data on user-item relationships for each user-company in format that ALS can use.
# We require each unique assignee ID in the rows of the matrix, and each unique item ID in columns of matrix.
# Values of matrix should be (?) binary user-item preference * confidence

# In[ ]:


import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pandas as pd
import numpy as np

from test_model import (get_patent_fields_list, get_ml_patents, 
                        create_title_abstract_col,trim_data, 
                        structure_dataframe, partition_dataframe, 
                        build_pipeline, process_docs, pat_inv_map, get_topics)

from rec_system import alphanum_to_int, int_to_alphanum, get_pat_recs

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary, mmcorpus
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.ldamodel import LdaModel
from gensim.models import AuthorTopicModel
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
import requests
from bs4 import BeautifulSoup
import pickle
import math

import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim

from pprint import pprint

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


# create Spark session
spark = pyspark.sql.SparkSession.builder.getOrCreate()
spark


# In[5]:


# create Spark context
sc = spark.sparkContext
sc


# ### Data understanding - Acquire data

# #### Data understanding - Acquire data for text workflows

# In[6]:


# load pickled dataset
with open('/Users/lee/Documents/techniche/techniche/data/raw_data_1000', 'rb') as f:
    raw_data_1000 = pickle.load(f)


# In[ ]:


# define keys as criteria to subset dataset #1 for non-text workflows
retained_keys = ['patent_number', 'patent_firstnamed_assignee_id']

# subset raw dataset by desired keys/columns
data_1000 = trim_data(data=raw_data_1000, keys=retained_keys)


# In[ ]:


# define keys as criteria to subset dataset #2, for text workflows
retained_keys_2 = ['patent_number', 'patent_firstnamed_assignee_id',
                   'patent_title', 'patent_abstract']

# subset raw dataset by desired keys/columns for text analysis workflows
data_1000_2 = trim_data(data=raw_data_1000, keys=retained_keys_2)


# #### Data preparation

# In[ ]:


# create new item in dataset #2 by concat of patent_title and patent_abstract
data_1000_2 = create_title_abstract_col(data=data_1000_2)


# In[ ]:


# create Pandas dataframe from dataset #1
df_1000 = pd.DataFrame(data_1000)


# In[ ]:


# create Pandas dataframe from dataset #2
df_1000_2 = pd.DataFrame(data_1000_2)
df_1000_2.head(3)


# In[ ]:


# for dataset #1: drop row that contains invalid data
df_1000[df_1000.patent_number.str.contains('[RE]')]
df_1000 = df_1000.drop(df_1000.index[[717]])

# drop NaNs in patent_firstnamed_assignee_id column
df_1000 = df_1000.dropna()


# In[ ]:


# for dataset#2: drop row that contains invalid data
df_1000_2[df_1000_2.patent_number.str.contains('[RE]')]
df_1000_2 = df_1000_2.drop(df_1000_2.index[[717]])

# drop NaNs in patent_firstnamed_assignee_id column
df_1000_2 = df_1000_2.dropna()


# #### Data preparation - model #1
# Prepare data on user-item relationships for each user-company in format that ALS can use.
# We require each unique assignee ID in the rows of the matrix, and each unique item ID in columns of matrix.
# Values of matrix should be (?) binary user-item preference * confidence

# In[ ]:


# create new rating column and assign value of 1
df_1000['rating'] = 1


# In[ ]:


# convert patent_number column from string to int
df_1000 = df_1000.astype({'patent_number': 'int64'})
# uncomment to confirm
# df_1000.info()


# In[ ]:


# convert alphanumeric patent_firstnamed_assignee_id col to int
df_1000 = df_1000.astype({'patent_number': 'int64'})


# In[ ]:


# df_1000['patent_firstnamed_assignee_id'] = df_1000['patent_firstnamed_assignee_id'].apply(hash).apply(abs)
df_1000['patent_firstnamed_assignee_id'] = df_1000['patent_firstnamed_assignee_id'].apply(hash).apply(abs) % 65536 # 2^16


# In[ ]:


# df_1000['patent_firstnamed_assignee_id'] = df_1000['patent_firstnamed_assignee_id'].apply(hash).apply(abs)
df_1000['patent_number'] = df_1000['patent_number'] % 65536 # 2^16


# In[ ]:


df_1000 = df_1000.astype({'patent_firstnamed_assignee_id': 'int'})


# #### Data preparation - model #1 - create Spark dataframe from pandas dataframe

# In[ ]:


sp_df_1000 = spark.createDataFrame(df_1000)


# In[ ]:


# cast columns from bigint to int
sp_df_1000_2 = sp_df_1000.withColumn("patent_firstnamed_assignee_id",
                                     sp_df_1000["patent_firstnamed_assignee_id"]
                                     .cast(IntegerType())).withColumn("patent_number",
                                     sp_df_1000["patent_number"]
                                     .cast(IntegerType())).withColumn("rating", 
                                     sp_df_1000["rating"].cast(IntegerType()))


# In[ ]:


# partition dataframe 
(training, test) = sp_df_1000.randomSplit([0.8, 0.2])


# ### Model # 1
# Build the recommendation model using ALS on the training data

# In[ ]:


# build ALS recommendation model
als = ALS(maxIter=5,
          regParam=0.01, 
          rank=10, # number of latent topics- ME-10?
          alpha=30,
          implicitPrefs=True, # # implicitPrefs=True b/c ratings are implicit
          userCol="patent_firstnamed_assignee_id", 
          itemCol="patent_number", 
          ratingCol="rating",
          coldStartStrategy="nan") # coldStartStrategy="nan" to retain NaNs


# In[ ]:


# fit ALS model to the training set
model = als.fit(training)


# #### Model #1 - Evaluation - Compare to naive baseline
# Compare model evaluation result with naive baseline model that only outputs (for explicit - the average rating (or you may try one that outputs the average rating per movie).

# #### Model #1 - Optimize model

# In[ ]:


# optimize model


# #### Getting Predictions

# In[ ]:


# get predictions for test set
predictions_test = model.transform(test)
predictions_test_df = predictions_test.toPandas()


# In[ ]:


# get predictions for training set
predictions_train = model.transform(training)
predictions_train_df = predictions_train.toPandas()
predictions_train_df


# In[ ]:


predictions_train_df.dropna()


# In[ ]:


predictions_test_df.dropna()


# ### Model #2 - Data preparation

# In[ ]:


train_text, test_text = train_test_split(df_1000_2, test_size = 0.2)


# #### Model 2 - Data preparation - text data

# - TF-IDF vectorization of patents - metrics - avg distance between individual patents, with ranking
# - take tf-idf vector and argsort by absolute value, to see which features are most important to patent
# - get top 20 features. normally would do cosine distance betweel all vectors. BUT, only do cosine distance between these top 20 features, for cold start patents

# In[ ]:


# instantiate TF-IDF Vectorizer using standard English stopwords
tfidf = TfidfVectorizer(stop_words='english')


# In[ ]:


# fit TF-IDF matrix on text column
tfidf_matrix = tfidf.fit_transform(train_text['patent_title_abstract'])


# In[ ]:


# output matrix, 972 docs, 5364 terms
tfidf_matrix.shape


# ### Model 3 - compute distance metric

# In[ ]:


# compute cosine similarity matrix between docs using linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[ ]:


# construct reverse map of indices and pat_title_abstract
indices = pd.Series(train_text.index, index = train_text['patent_number']).drop_duplicates()


# In[ ]:


# tfidf vec requires list, not just string
unseen_data = test_text
unseen_data


# In[ ]:


unseen_tfidf = tfidf.transform(unseen_data['patent_title_abstract'])
unseen_tfidf.shape


# In[ ]:


# pass patent number from training set to get_recommendations
# take user input of string and output most similar documents
get_pat_recs('10019674')


# In[ ]:


train_text.head()


# #### Model #2 - Apply K means clustering to distance matrix

# In[ ]:


km = KMeans(20)


# In[ ]:


kmresult = km.fit(tfidf_matrix).predict(unseen_tfidf)


# In[ ]:


kmresult_p = km.predict(unseen_tfidf)


# In[ ]:


kmresult_p


# In[ ]:




