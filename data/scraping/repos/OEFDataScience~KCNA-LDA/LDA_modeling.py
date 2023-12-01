# -*- coding: utf-8 -*-
"""
LDA Topic Modeling Training and pre-canned dashboard creation
k=10
Created on Wed Apr 29 10:56:40 2020
@author: Claytonious
"""
#modules
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.test.utils 
import spacy
import en_core_web_sm
import nltk 
#nltk.download('stopwords')
from nltk.corpus import stopwords
import tqdm
import re
import numpy as np
import pickle
########################
#utility functions
def remove_stopwords(texts):
    return[[word for word in simple_preprocess(str(doc)) if word not in stop_words]
            for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB',
                                          'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

##################################

#DATA IMPORT AND SETUP
df2 = pd.read_pickle("dprk_cleaned.pkl")
df2['year'] = pd.DatetimeIndex(df2['Date']).year
years = range(1996, 2020)


###########################################
#LDA w/ GENSIM FRAMEWORK
#part 1: setup bigrams/trigrams
#################################################
#bigrams and trigram phrase models
###################################################
bigram = gensim.models.Phrases(df2['text_processed2'], min_count = 5,
                               threshold = 100)

trigram = gensim.models.Phrases(bigram['text_processed2'],
                                threshold = 100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
#####################################################
#part 2: Remove stopwords and lemmatize text
######################################################
#stopword model setup
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
#remove stopwords
data_words_nostops = remove_stopwords(df2['text_processed2'])
#form bigram
data_words_bigram = make_bigrams(data_words_nostops)
#initialize spacy 'en' model
nlp = en_core_web_sm.load(disable = ['parser', 'ner'])
#lemmatization, keeping only noun, adj, vb, adv
data_lemma = lemmatization(data_words_bigram, 
                           allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#######################################################
#part 3: Train LDA model 
#######################################################
#model object setup
#dictionary
id2word = corpora.Dictionary(data_lemma)
#save dictionary
id2word.save("id2word.pkl")
#corpus
texts = data_lemma
#term document matrix
corpus = [id2word.doc2bow(text) for text in texts]
#save DTM/corpus
with open('corpus.pkl', "wb") as fp:
    pickle.dump(corpus, fp)
#train model with k=10, random seed = 100
lda_model = gensim.models.LdaMulticore(corpus = corpus,
                                       id2word = id2word,
                                       num_topics = 10,
                                       random_state = 100,
                                       chunksize = 100,
                                       passes = 10,
                                       per_word_topics=True
#save lda_model                                       
lda_model.save("lda_10.pkl")
#lda_model = gensim.models.LdaModel.load("lda_baseline.pkl")

#inspect for coherence score
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
#coherence
coherence_model_lda = CoherenceModel(model=lda_model,
                                     texts = data_lemma,
                                     dictionary = id2word,
                                     coherence = 'c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
#baseline coherence score is 0.5407 with k = 10

#create dashboard visualization hosted on github repo page
import pyLDAvis.gensim
import pickle
import pyLDAvis
import os

#first visualize the 10 topic baseline model
LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(10))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:

    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)


with open(LDAvis_data_filepath, 'w') as f:
        pickle.dump(LDAvis_prepared, f)
        
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath) as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(10) +'.html')




