import pandas as pd
import numpy as np
import gensim
import nltk
import logging
import pickle
import spacy
import math
import warnings
import sys
sys.path.append('qtester')
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gensim.corpora as corpora

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
from gensim.test.utils import datapath

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from scipy.stats import entropy
from tempfile import TemporaryFile

from scipy.special import (entr, rel_entr)
from numpy import (arange, putmask, ravel, ones, shape, ndarray, zeros, floor,
                   logical_and, log, sqrt, place, argmax, vectorize, asarray,
                   nan, inf, isinf, NINF, empty)

from libs.my_paths import base_model_lda, base_model_ngram, base_model

MY_STOP_WORDS = STOPWORDS.union(set(['use', 'be', 'work', 'user', 'try', 'cell',
                                     'row', 'want', 'item', 'go', 'get', 'add', 'went', 'tried',
                                     'return', 'sort', 'test', 'run', 'check', 'click', 'hour', 'minute', 'second',
                                     'version', 'app', 'paragraph', 'error', 'log', 'press',
                                     'need', 'feed', 'thank', 'way', 'like', 'kill', 'help']))

def clear_text(text):
    text = re.sub('<code>(.|\n)*?<\/code>', '', text)
    text = re.sub(r'(\<(/?[^>]+)>)', '', text)
    text = re.sub("[\'\"\\/\@\%\(\)\~\`\{\}]", '', text)
    text = re.sub('\s+', ' ', text)
    
    return text

def lemmatize_stemming(text, stemmer):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    text = clear_text(text)
    result = []
    #result = [token in gensim.utils.simple_preprocess(text, deacc=True) if ((token not in gensim.parsing.preprocessing.STOPWORDS) and len(token) > 1) == True]
    for token in gensim.utils.simple_preprocess(text, deacc=True):
        if (token not in MY_STOP_WORDS) and len(token) > 1:
            #result.append(lemmatize_stemming(token))
            result.append(token)
    return result

def split_tags(text):
    if not isinstance(text, str) and math.isnan(text):
        return ''
    if text == '' or text == ' ':
        return text
    else:
        return text.replace('|', ' ')

def add_string(text, tags, n=3):
    tags = split_tags(tags)
    tags = ' ' + tags
    i = 0
    for i in range(n):
        if i % 2 == 0:
            text += tags
        else:
            text = tags + text
    return text

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags and not token.lemma_ in MY_STOP_WORDS])
    return texts_out

def get_text_bow(text, dictionary, bigram_mod, trigram_mod, nlp):
    text = preprocess(text)
    text = make_bigrams([text], bigram_mod)[0]
    text = make_trigrams([text], bigram_mod, trigram_mod)[0]
    text = lemmatization([text], nlp)[0]
    bow_vector = dictionary.doc2bow(text)
    return bow_vector

def test_texts(text1, text2, lda_model):
    bow1 = get_text_bow(text1)
    bow2 = get_text_bow(text2)
    sc1 = 0.0
    sc2 = 0.0
    for index, score in sorted(lda_model[bow1], key=lambda tup: -1*tup[1]):
        print(f"index: {index}, score {score}")
        sc1 += score
        #print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    print("_________________________________________")
    for index, score in sorted(lda_model[bow2], key=lambda tup: -1*tup[1]):
        print(f"index: {index}, score {score}")
        sc2 += score
    return (sc1, sc2)
        
def jensen_shannon_v(p, q):
    p = p[None,:].T
    q = q[None,:].T
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))


def distr(arr):
    max_len = len(arr)
    mat = []
    mat.append([1.0 for i in range(max_len)])
    for i in range(max_len - 1):
        z = [0.0 for k in range(max_len)]
        z[i] = arr[i + 1]
        z[i + 1] = -arr[i]
        mat.append(z)
    vec = np.zeros(max_len)
    vec[0] = 1.0
    mat = np.array(mat)
    print(mat)
    print(vec)
    #return np.linalg.solve(mat, vec)


class TextProcessor():
    def __init__(self):
        self._lda_model = gensim.models.LdaModel.load(datapath(base_model_lda + 'model_semi_final'))
        self._dictionary = gensim.models.LdaModel.load(datapath(base_model_lda + 'model_semi_final.id2word'))
        self._bigram_mod = gensim.models.LdaModel.load(datapath(base_model_ngram + 'bigram_mod'))
        self._trigram_mod = gensim.models.LdaModel.load(datapath(base_model_ngram + 'trigram_mod'))
        self._nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        np.random.seed(2020)

    def title_body_sim(self, text1: str, text2: str, tags=None, n_topics: int = -1) -> float:
        if tags != None:
            text1 = add_string(text1, tags)
            text2 = add_string(text2, tags)

        bow1 = get_text_bow(text1, self._dictionary, self._bigram_mod, self._trigram_mod, self._nlp)
        bow2 = get_text_bow(text2, self._dictionary, self._bigram_mod, self._trigram_mod, self._nlp)

        if n_topics == -1:
            n_topics = self._lda_model.num_topics

        p = np.zeros(n_topics)
        q = np.zeros(n_topics)

        for index, score in sorted(self._lda_model[bow1], key=lambda tup: -1*tup[1]):
            p[index] = score
        for index, score in sorted(self._lda_model[bow2], key=lambda tup: -1*tup[1]):
            q[index] = score
        jsd = jensen_shannon_v(p, q)[0]
        if (math.isnan(jsd)):
            jsd = 0.0
        return jsd

    def get_lda(self, text: str, n_topics: int = -1) -> np.array:
        bow = get_text_bow(text, self._dictionary, self._bigram_mod, self._trigram_mod, self._nlp)

        if n_topics == -1:
            n_topics = self._lda_model.num_topics

        p = np.zeros(n_topics)
        for index, score in sorted(self._lda_model[bow], key=lambda tup: -1*tup[1]):
            p[index] = score
        return p

