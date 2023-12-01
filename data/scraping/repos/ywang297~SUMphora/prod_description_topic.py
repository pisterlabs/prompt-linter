# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:39:47 2018

@author: yishu
"""

import pickle 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('wordnet')

productDescriptions=pd.read_pickle("/Users/yishu/Documents/insight/productDescriptions_raw_and_cleaned.p")

prod_des = productDescriptions[['p_description_clean']]
prod_des.dropna(inplace=True)

prod_des['index']=productDescriptions.index

documents =prod_des

documents[:5]

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

## example 

#np.random.seed(2018)

#print(WordNetLemmatizer().lemmatize('went', pos='v'))

stemmer = SnowballStemmer('english')

#original_words = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 
#           'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 
#           'traditional', 'reference', 'colonizer','plotted']
#singles = [stemmer.stem(plural) for plural in original_words]
#pd.DataFrame(data = {'original word': original_words, 'stemmed': singles})

#def  lemmatize_stemminglemmati (text):
#    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

stop_words = list(gensim.parsing.preprocessing.STOPWORDS)
#stop_words.extend(['like', 'love', 'good', 'tried', 'great'])

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stop_words and len(token) > 3:
            result.append(token)
    return result

#doc_sample = documents[documents['index'] == 4310].values[0][0]


#print('original document: ')
#words = []
#for word in documents.split(' '):
#    words.append(word)
#print(words)
#print('\n\n tokenized and lemmatized document: ')
#print(preprocess(documents))

processed_docs = documents['p_description_clean'].map(preprocess)
processed_docs[:10]

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[0]

from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
#Topic: 0 
#Words: 0.039*"brush" + 0.023*"makeup" + 0.014*"lashes" + 0.012*"fibers" + 0.012*"lash" + 0.011*"mascara" + 0.011*"application" + 0.011*"brushes" + 0.010*"fiber" + 0.009*"collection"
#Topic: 1 
#Words: 0.014*"color" + 0.013*"brow" + 0.012*"formula" + 0.011*"dior" + 0.008*"brows" + 0.008*"pencil" + 0.007*"brush" + 0.007*"rose" + 0.006*"long" + 0.006*"hours"
#Topic: 2 
#Words: 0.015*"product" + 0.013*"brush" + 0.012*"free" + 0.010*"tested" + 0.009*"lips" + 0.008*"women" + 0.007*"soft" + 0.007*"formula" + 0.006*"mask" + 0.006*"appearance"
#Topic: 3 
#Words: 0.016*"makeup" + 0.010*"free" + 0.010*"parabens" + 0.008*"tested" + 0.008*"product" + 0.007*"fragrance" + 0.007*"looking" + 0.007*"look" + 0.006*"formula" + 0.006*"phthalates"
#Topic: 4 
#Words: 0.014*"natural" + 0.012*"makeup" + 0.009*"oily" + 0.009*"finish" + 0.009*"pores" + 0.009*"combination" + 0.008*"foundation" + 0.008*"free" + 0.008*"formula" + 0.008*"look"
#Topic: 5 
#Words: 0.013*"fresh" + 0.012*"formula" + 0.009*"product" + 0.009*"tested" + 0.009*"free" + 0.009*"parabens" + 0.008*"natural" + 0.008*"moisture" + 0.008*"water" + 0.007*"oily"
#Topic: 6 
#Words: 0.011*"color" + 0.009*"cream" + 0.009*"shades" + 0.009*"palette" + 0.008*"natural" + 0.008*"formula" + 0.007*"coverage" + 0.007*"product" + 0.007*"brush" + 0.006*"look"
#Topic: 7 
#Words: 0.013*"brush" + 0.010*"makeup" + 0.010*"type" + 0.009*"texture" + 0.009*"want" + 0.009*"good" + 0.009*"combination" + 0.009*"ingredients" + 0.009*"sensitive" + 0.009*"solutions"
#Topic: 8 
#Words: 0.015*"look" + 0.012*"formula" + 0.010*"color" + 0.010*"lines" + 0.009*"fine" + 0.009*"dark" + 0.009*"cream" + 0.008*"application" + 0.008*"product" + 0.007*"texture"
#Topic: 9 
#Words: 0.012*"hair" + 0.012*"mask" + 0.010*"color" + 0.009*"formula" + 0.009*"texture" + 0.008*"product" + 0.007*"natural" + 0.007*"face" + 0.006*"lips" + 0.006*"long"    

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
#Topic: 0 
#Words: 0.006*"mask" + 0.004*"masks" + 0.003*"women" + 0.003*"cream" + 0.003*"texture" + 0.003*"dryness" + 0.003*"tone" + 0.003*"lines" + 0.003*"wrinkles" + 0.003*"visibly"
#Topic: 1 
#Words: 0.005*"makeup" + 0.005*"color" + 0.004*"lips" + 0.004*"shades" + 0.003*"primer" + 0.003*"foundation" + 0.003*"pencil" + 0.003*"scent" + 0.003*"balm" + 0.003*"wear"
#Topic: 2 
#Words: 0.005*"brow" + 0.005*"color" + 0.004*"lips" + 0.004*"black" + 0.004*"shades" + 0.004*"matte" + 0.004*"notes" + 0.004*"lash" + 0.004*"lashes" + 0.003*"palette"
#Topic: 3 
#Words: 0.007*"kiehl" + 0.006*"ingredients" + 0.005*"quality" + 0.004*"brush" + 0.004*"rich" + 0.004*"protective" + 0.004*"natural" + 0.004*"finest" + 0.004*"efficacious" + 0.004*"hair"
#Topic: 4 
#Words: 0.010*"brush" + 0.006*"hair" + 0.004*"makeup" + 0.004*"brushes" + 0.004*"eyeliner" + 0.004*"color" + 0.004*"fibers" + 0.004*"fiber" + 0.004*"palette" + 0.004*"straight"
#Topic: 5 
#Words: 0.005*"hair" + 0.005*"brush" + 0.004*"color" + 0.004*"makeup" + 0.003*"lips" + 0.003*"fresh" + 0.003*"cleansing" + 0.003*"liner" + 0.003*"free" + 0.003*"clean"
#Topic: 6 
#Words: 0.007*"mascara" + 0.007*"lashes" + 0.006*"lash" + 0.005*"sodium" + 0.005*"makeup" + 0.005*"volume" + 0.005*"brush" + 0.004*"foundation" + 0.004*"free" + 0.004*"product"
#Topic: 7 
#Words: 0.004*"wrinkles" + 0.004*"looking" + 0.004*"coverage" + 0.004*"said" + 0.004*"cream" + 0.004*"foundation" + 0.004*"complexion" + 0.004*"appearance" + 0.004*"makeup" + 0.004*"oily"
#Topic: 8 
#Words: 0.013*"brush" + 0.005*"makeup" + 0.005*"application" + 0.005*"brushes" + 0.004*"lashes" + 0.004*"mascara" + 0.004*"fibers" + 0.004*"dark" + 0.003*"fiber" + 0.003*"color"
#Topic: 9 
#Words: 0.004*"powder" + 0.003*"cleansing" + 0.003*"makeup" + 0.003*"soap" + 0.003*"glow" + 0.003*"pores" + 0.003*"mask" + 0.003*"color" + 0.003*"complexion" + 0.003*"brush"
    
# Compute Perplexity for lda_model
print('Perplexity: ', lda_model.log_perplexity(bow_corpus))  # a measure of how good the model is. lower the better.
#Perplexity:  -6.812367467593914

# Compute Coherence Score for lda_model
from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)
#Coherence Score:  0.3877288722955065

# Compute Perplexity
print('Perplexity: ', lda_model_tfidf.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.
#Perplexity:  -9.054514994363373

# Compute Coherence Score
from gensim.models import CoherenceModel
coherence_model_lda_tfidf = CoherenceModel(model=lda_model_tfidf, texts=processed_docs, dictionary=dictionary, coherence='c_v')
coherence_lda_tfidf = coherence_model_lda_tfidf.get_coherence()
print('Coherence Score: ', coherence_lda_tfidf)
#Coherence Score:  0.41254989458223995    






    