#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:18:54 2022

@author: dataguy
"""

# Import libraries
import os
import pandas as pd
import re
import string
import nltk
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, ArrayType, FloatType
from pyspark.sql.functions import udf, col, size, lit, explode, isnan, when, count, min, max, struct
import matplotlib.pyplot as plt
from textblob import TextBlob

# Init language
words = set(nltk.corpus.words.words())
stop_words = stopwords.words('english')

# Function to get hashtags
def extract_hashtags(x):
    
    hashtag_list = []
      
    # splitting the text into words
    for word in x.split():
          
        # checking the first charcter of every word
        if word[0] == '#':
              
            # adding the word to the hashtag_list
            hashtag_list.append(word[1:])
      
    return hashtag_list

stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())

# Function to filter
def process_txt(x):
    
    x = x.translate(str.maketrans('', '', string.punctuation))
    x = re.sub('\d+', '', x).lower().split()
    x = [i for i in x if i in words]
    x = [i for i in x if i not in stop_words and len(i) > 3]
    
    return x

# Function to tag grammar
def filter_pos(x):
    
    tags = nltk.pos_tag(x)
    x1 = [x[0] for x in tags if x[0] in words and len(x[0]) > 3 and x[1].startswith(('N', 'J', 'V'))]

    if len(x1) > 0:
        
        return ' '.join(x1)
    
    else:
        
        return None

# Function to label tone
def get_tone(score):
    
    if (score >= 0.1):
        
        label = "positive"
   
    elif (score <= -0.1):
        
        label = "negative"
        
    else:
        
        label = "neutral"
        
    return label