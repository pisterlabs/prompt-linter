#!/usr/bin/env python3

#### BASED ON ##################################################################
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

#### DEPENDENCIES ##############################################################
# pip3 install gensim matplotlib nltk numpy pandas pyLDAvis spacy
# python3 -m spacy.en.download
# python3 -m nltk.downloader stopwords

#### PYTHON IMPORTS ############################################################
import re
import numpy as NP
import pandas as PD
from pprint import pprint

#### GENSIM IMPORTS ############################################################
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#### SPACY IMPORTS #############################################################
import spacy

#### VISUALIZATION IMPORTS #####################################################
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as PLT

#### META IMPORTS ##############################################################
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#### NLTK IMPORTS ##############################################################
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

#### DATA PREPARATION ##########################################################
df = PD.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
print(df.target_names.unique())
df.head()

# Convert to list
data = df.content.values.tolist()
# Remove email addresses
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
# Remove '\n' and '\r'
data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove single quotes
data = [re.sub("\'", "", sent) for sent in data]

pprint(data[:1])

#### TOKENIZATION ##############################################################
def sentToTokens(sents):
    for sent in sents:
        # `deacc=True` strips punctuation
        yield(gensim.utils.simple_preprocess(str(sent), deacc=True))

data_tokens = list(sentToTokens(data))
# print(data_tokens[:1])

#### N-GRAM MODELS #############################################################
# A higher threshold results in fewer phrases
bigrammer_init = gensim.models.Phrases(data_tokens, min_count=5, threshold=100)
trigrammer_init = gensim.models.Phrases(bigrammer_init[data_tokens], threshold=100)

bigrammer = gensim.models.phrases.Phraser(bigrammer_init)
trigrammer = gensim.models.phrases.Phraser(trigrammer_init)

# print(trigrammer[bigrammer[data_tokens[0]]])

#### TEXT NORMALIZATION (CLEANING) #############################################
def removeStopwords(texts):
    return [
        [w for w in simple_preprocess(str(doc)) if w not in stop_words]
        for doc in texts
    ]

def makeBigrams(texts):
    return [bigrammer[doc] for doc in texts]

def makeTrigrams(texts):
    return [trigrammer[bigrammer[doc]] for doc in texts]

def lemmatize(texts, allowed_pos=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = list()
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([tok.lemma_ for tok in doc if tok.pos_ in allowed_pos])
    return texts_out

data_tokens_clean = removeStopwords(data_tokens)
data_bigrams = makeBigrams(data_tokens_clean)
nlp = spacy.load('en', disable=['parser', 'ner'])
data_lemmas = lemmatize(data_bigrams)

print(data_lemmas[:1])

#### BUILD CORPUS ##############################################################
id_to_token_map = corpora.Dictionary(data_lemmas)
texts = data_lemmas
tf_matrix = [id_to_token_map.doc2bow(text) for text in texts]

# print(tf_matrix[:1])
# print(id_to_token_map[0])

# human_tf_matrix = [[(id_to_token_map[i], freq) for i, freq in doc] for doc in tf_matrix]

#### BUILD TOPIC MODEL #########################################################
# DOCS: https://radimrehurek.com/gensim/models/ldamodel.html
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=tf_matrix, # term-frequency matrix
    id2word=id_to_token_map, # map from token id to actual token
    num_topics=20, # how many latent topics to be extracted
    random_state=100, # for reproduceability
    update_every=1, # read docs
    chunksize=100, # number of documents to use in each training chunk
    passes=10, # number of passes through corpus during training
    alpha='auto', # read docs
    per_word_topics=True # read docs
)

#### RESULTS ###################################################################
# `lda_model.print_topics()` returns a list of size `num_topics` (from previous)
# where each element is a tuple containing an index of the topic and a
# combination of keywords associated with that topic and their respective
# weights in the topic. For example:
#
# lda_model.print_topics()[0]
#  (0,
#    '0.016*"car" + 0.014*"power" + 0.010*"light" + 0.009*"drive" +
#     0.007*"mount" + 0.007*"controller" + 0.007*"cool" + 0.007*"engine" +
#     0.007*"back" + 0.006*"turn"'
#  )
#
# The output above describes a topic that could be described as 'cars', etc.
# where 'car' is the dominating term with a weight of 0.016.
pprint(lda_model.print_topics())
doc_lda = lda_model[tf_matrix]

#### EVALUATION ################################################################
# Perplexity
print('\nPerplexity: {}'.format(lda_model.log_perplexity(tf_matrix)))

# Coherence Score: https://radimrehurek.com/gensim/models/coherencemodel.html
coherence = CoherenceModel(
    model=lda_model, # the model we care about
    texts=data_lemmas, # the lemmas in the model
    dictionary=id_to_token_map, # mapping of token id to actual token
    coherence='c_v' 
)
coherence = coherence.get_coherence()
print('\nCoherence: {}'.format(coherence))

#### VISUALIZATION #############################################################
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, tf_matrix, id_to_token_map)
# vis
