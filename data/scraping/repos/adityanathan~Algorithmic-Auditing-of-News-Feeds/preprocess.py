import pandas as pd
import numpy as np
import pickle
import re
import timeit
import spacy

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, HdpModel, LdaModel, LdaMulticore
from nltk.corpus import stopwords
import helper as he
stop_words = stopwords.words('english')

'''
Preprocess your data using this code and save it 
in a pickle file so you can reuse it later.
'''

input_file = 'data/massmedia-data/farmers_date_indexed.pkl'
output_file = 'data/preprocessed_data/corpus_dict/farmers_corp.pkl'

print('Loading Documents.....')
documents = []
with open(input_file, 'rb') as f:
    docs,_ = pickle.load(f)
    for i in docs:
        documents.append(i['text'])
        
print('Sample Document')
print(documents[250])

print('Simple Preprocessing')
# Document Preprocessing
data = documents.copy()
# Removes phrases with @ in them
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
# Truncates multiple consecutive whitespace to one
data = [re.sub('\s+', ' ', sent) for sent in data]
# Removes ' characters
data = [re.sub("\'", "", sent) for sent in data]

data_words = list(he.sent_to_words(data))
print('Building Bigrams')
# Making Bigrams - Higher the threshold, fewer the phrases
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
print('Removing Stopwords')
# Remove Stop Words
data_words_nostops = he.remove_stopwords(data_words, stop_words)
print('Forming Bigrams')
# Form Bigrams
data_words_bigrams = he.make_bigrams(data_words_nostops, bigram_mod)
print('Lemmatizing Data')
# Lemmatize Data
data_lemmatized = he.lemmatization(data_words_bigrams, allowed_postags=[
    'NOUN', 'ADJ', 'VERB', 'ADV'])


# The keep_n parameter controls the size of the vocabulary.
# At this stage, we have to manually experiment with various vocabulary sizes to see what works best.
# I found that ~8-10% of the number of documents is a good size.
# For Digital India, I used vocab size of 1000 (12412 documents).
# For GST, I used a vocab size of 1500 (15k documents approx)

print('Creating Dictionary')
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Filter words
id2word.filter_extremes(no_below=5, no_above=0.95,
                        keep_n=1800, keep_tokens=None)

# Lemmatized data is your corpus

print('Converting corpus using dictionary')
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_lemmatized]

# Save Data in pickle file
with open(output_file, 'wb') as f:
    pickle.dump((data_lemmatized, id2word, corpus), f)
