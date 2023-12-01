#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:30:07 2022

@author: dataguy
"""

# Import libraries
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, ArrayType, FloatType
import re
import string
from pyspark.sql.functions import udf, col, size, lit, explode, isnan, when, count, min, max, struct
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, IndexToString, StringIndexer, VectorIndexer, CountVectorizer
from collections import Counter
import networkx as nx
import nltk
from nltk.corpus import words
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
import matplotlib.pyplot as plt
%matplotlib inline
import openai
import json
import ast
from igraph import *
import datashader as ds
import datashader.transfer_functions as tf
from datashader.layout import random_layout, circular_layout, forceatlas2_layout
from datashader.bundling import connect_edges, hammer_bundle
from textblob import TextBlob
import numpy as np
import os
os.chdir('/home/dataguy/Documents/')
from tweetology import establish_conn

'''
    Get raw data
'''

# Establish connection
conn = establish_conn(uri = "bolt://44.204.227.162:7687", user = "neo4j", pwd = "discards-vices-stator")

# Initialize spark
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.getOrCreate()

# Read
df = spark.read.json('/home/dataguy/data/us_news_tweets.json')

# Inspect
min_date, max_date = df.select(min("date"), max("date")).first()
print(f'Total # Tweets: {df.count()}')
print(f"Min Date- {min_date}")
print(f"Max Date- {max_date}")

# Expand
df1 = df.withColumn('username', col('user.username')).withColumn('country', col('place.country')).withColumn('country_cd', col('place.countryCode'))
df2 = df1.withColumn('quoted', col('quotedTweet.user.username')).withColumn('mentions', col('mentionedUsers.username')).withColumn('reply_to', col('inReplyToUser.username')).drop('inReplyToUser', 'mentionedUsers').filter("country == 'United States'").toPandas().drop(['renderedContent', 'id', 'media', 'outlinks', '_type', 'cashtags', 'conversationId', 'inReplyToTweetId', 'source', 'sourceUrl', 'sourceLabel', 'tcooutlinks', 'url', 'country', 'lang', 'retweetedTweet', 'user', 'coordinates', 'place', 'quotedTweet'], axis = 1)
df3 = df2.set_index(pd.DatetimeIndex(df2['date'])).drop('date', axis = 1)

# Separate users
nodes = [i for i in df3['username'].unique()]

# Add user activites
for i in df3['quoted']:
    
    if i is not None:
        
        nodes.append(i)
        
for x in df3['mentions']:
    
    if x is not None:
        
        for y in x:
            
            nodes.append(y)
            
for z in df3['reply_to']:
    
    if z is not None:
        
        nodes.append(z)
        
# Build the edges data
edges = []

replies = df3.dropna(subset = ['reply_to'])
mentions = df3.dropna(subset = ['mentions'])
quotes = df3.dropna(subset = ['quoted']) 

# Iterate the data to add replies
for idx, row in replies.iterrows():
    
    edges.append((row['username'], row['reply_to'], 'reply'))
    
# Iterate the data to add mentions
for idx, row in mentions.iterrows():
    
    for entity in row['mentions']:
        
        edges.append((row['username'], entity, 'mention'))
        
# Iterate the data to add quotes
for idx, row in quotes.iterrows():
    
    edges.append((row['username'], row['quoted'], 'quote'))

# Convert
edges = pd.DataFrame(edges, columns = ['source', 'target', 'type'])
edges = edges[edges['source'] != edges['target']]

# Calculate edge weights
weights = edges.groupby(['source', 'target', 'type']).size().reset_index().rename({0: 'weight'}, axis = 1)

'''
    Build graph
'''

# Subset
top = weights.sort_values('weight', ascending = False)[0:100]

# Init graph
conn.query('create or replace database usnews')

# Add relationships to graph
types = pd.DataFrame(weights['type'].unique(), columns = ['type'])

interactions = '''
                  unwind $rows as row 
                  merge (c: category {type: row.type})  
                  return count(*) as total
              '''
       
res = conn.query(interactions, db = 'usnews', parameters = {'rows': types.to_dict('records')})
print(f"Successfully processed {res[0]['total']} records..")

# Add subset to graph
users = '''
            unwind $rows as row
            merge (n: node {username: row.username})
            return count(*) as total
         '''

nodes = pd.DataFrame(top['source'].unique(), columns = ['username'])
new_nodes = pd.DataFrame(top['target'].unique(), columns = ['username'])

all_nodes = pd.concat([nodes, new_nodes])
all_nodes = pd.DataFrame(all_nodes['username'].unique(), columns = ['username'])

res = conn.query(users, db = 'usnews', parameters = {'rows': all_nodes.to_dict('records')})
print(f"Successfully processed {res[0]['total']} records..")

# Add tweets to graph
edges = '''
              unwind $rows as row
              match (source: node {username: row.source}),
                    (target: node {username: row.target})
              create (source) - [:type] -> (target)
              return count(*) as total
        '''

res = conn.query(edges, db = 'usnews', parameters = {'rows': top.to_dict('records')})    
print(f"Successfully processed {res[0]['total']} records..")

