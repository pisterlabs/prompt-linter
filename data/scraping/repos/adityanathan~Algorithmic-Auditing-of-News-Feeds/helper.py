import pandas as pd
import numpy as np
import pickle
import re
import spacy

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, HdpModel
from nltk.corpus import stopwords

'''
A collection of helper functions
'''

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# Define functions for stopwords, bigrams, trigrams and lemmatization


def remove_stopwords(texts, stop_words):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def create_dictionary(data_lemmatized):
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Filter words
    id2word.filter_extremes(no_below=5, no_above=0.95,
                            keep_n=1800, keep_tokens=None)
    return id2word


def create_model_corpus(id2word, data_lemmatized):
    return [id2word.doc2bow(text) for text in data_lemmatized]


def build_hdp(corpus, id2word):
    hdpmodel = HdpModel(corpus=corpus, id2word=id2word, chunksize=2000)
    hdptopics = hdpmodel.show_topics(formatted=False)
    hdptopics = [[word for word, prob in topic]
                 for topicid, topic in hdptopics]
    return hdpmodel, hdptopics
