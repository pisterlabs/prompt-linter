# Imports
# -*- coding: utf-8 -*-
import nltk#; nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
# Initialize spacy 'en' model, (only POS tagger component) (for speed)
# python3 -m spacy download en or python - m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# 1: Import our Corpus
df = pd.read_json('https://raw.githubusercontent.com/kudacall/nlpTutorial/master/newsgroups.json')
print(df.target_names.unique()) #Examine our topics
df.head() 

# 1.1: Clean up and format our corpus for our processing through NLP Pipeline
# Convert to list
data = df.content.values.tolist()
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]
# Quick check
# pprint(data[:1])

# 2: Use Gensim utilities to tokenize sentences and remove punctuation
def sentToWords(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(unicode(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sentToWords(data))
#Check tokens
# print(data_words[:1])

# 3: Tag tokens with POS tags
def tagTokenLists(tokenLists): #POS Tagging with NLTK
    for tokens in tokenLists:
        yield nltk.pos_tag(tokens)
#Check tags
taggedWords = tagTokenLists(data_words)
# print(next(taggedWords))

def lemmatize(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #Lemmatization and POS Tagging and filtering with SpaCy
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# 4: Remove stopwords
# NLTK Stop words
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
stopWords.extend(['from', 'subject', 're', 'edu', 'use'])
def removeStopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopWords] for doc in texts]
# Remove Stop Words
dataWords = removeStopwords(data_words)
dataLemmatized = lemmatize(dataWords)
#Check
# print(dataLemmatized[:1])

#5: Entity Recognition
def getEntsPG(object):
    from polyglot.text import Text
    text = Text(object)
    pgOut = []
    for sent in text.sentences:
        for entity in sent.entities:
            pgOut.append((entity.tag, entity))
    return pgOut

def getEntsTB(object):
    from textblob import TextBlob
    tbObject = TextBlob(object)
    return tbObject.noun_phrases

def getEntsSp(object):
    doc = nlp(object)
    return doc.ents

#6: Modeling
# Create Dictionary
id2word = corpora.Dictionary(dataLemmatized)
# Create Corpus
texts = dataLemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

# Build LDA model
print "Building LDA Model..."
ldaModel = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=20, random_state=100,
                                           update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)

#7: Check and visualize 
print "Building Visualization..."
# doctopic = ldaModel.get_topics()
# pprint(ldaModel.print_topics())
# pyLDAvis.enable_notebook() #enable if using Jupyter notebook
vis = pyLDAvis.gensim.prepare(ldaModel, corpus, id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')
