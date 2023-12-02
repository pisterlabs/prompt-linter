#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:39:49 2019

@author: Padmanabhan Rajendrakumar
"""


from pyspark.sql import SparkSession
import gensim
import gensim.corpora as corpora
from pprint import pprint
from gensim.models import CoherenceModel




# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import ast

spark = SparkSession \
    .builder \
    .appName('Topic Modelling - Cleaning - NYT') \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/nyt_db.nyt_coll") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/nyt_db.nyt_coll") \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

print("Reading from MongoDB...")
nyt_df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

print("Number of articles:",nyt_df.count())

data = nyt_df.toPandas()
doc_clean = data['tokens_final']
doc_clean = map(str, doc_clean) #For mongo
doc_clean1 = [ast.literal_eval(doc) for doc in doc_clean]

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean1)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean1]

# Creating the object for LDA model using gensim library
Lda = gensim.models.LdaMulticore

# Load a potentially pretrained model from disk.
ldamodel = gensim.models.LdaMulticore.load("ldamodel_nyt")
print("Top 25 Topics(LDA): ")
pprint(ldamodel.print_topics(num_topics=15, num_words=5))


lsimodel = gensim.models.LsiModel.load("lsimodel_nyt")
print("Top 25 Topics(LSI): ")
pprint(lsimodel.print_topics(-1))


#Perplexity
print('\nPerplexity: ', ldamodel.log_perplexity(doc_term_matrix))

# Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=doc_clean1, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
