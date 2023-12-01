# -*- coding: utf-8 -*-
"""
Guided_LDA
Follow this blog post
https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164
"""

from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel 
import numpy as np
import sys
import os
import gensim
import pickle
#from collections import Counter
#import pyLDAvis
#import pyLDAvis.gensim  # don't skip this
#import matplotlib
import matplotlib.pyplot as plt

python_root = './scripts'
sys.path.insert(0, python_root)

#%%
def prepare_data(save=True):
    ## read and transform data 
    contents = pickle.load(open('../data/lemma_corpus.p', "rb"))
    docs = list()
    for paragraph in contents:
        docs.append([w for sentance in paragraph for w in sentance])

    # build dictionary
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=5,no_above=0.5, keep_n=20000)
    # convert document into bow
    corpus_bow = [dictionary.doc2bow(text) for text in docs]
    ## comput tfidf feature vectors
    tfidf = models.TfidfModel(corpus_bow) # smartirs = 'atc' https://radimrehurek.com/gensim/models/tfidfmodel.html
    corpus_tfidf = tfidf[corpus_bow]
    
    ## save dictionary and corpora 
    if save:
        dictionary_save_path = '../data/dictionary.dict'
        dictionary.compactify()
        dictionary.save(dictionary_save_path)
        corpora.MmCorpus.serialize('../data/corpus_bow.mm', corpus_bow)
        corpora.MmCorpus.serialize('../data/corpus_tfidf.mm', corpus_tfidf)
        #print(len(dictionary))
    
    return docs,dictionary,corpus_bow,corpus_tfidf

#%%

## a better way to print 
def print_topics_gensim(topic_model, total_topics=1,
                        weight_threshold=0.0001,
                        display_weights=False,
                        num_terms=None):
    
    for index in range(total_topics):
        topic = topic_model.show_topic(index,topn=num_terms)
        topic = [(word, round(wt,4)) 
                 for word, wt in topic 
                 if abs(wt) >= weight_threshold]
        if display_weights:
            print('Topic #'+str(index+1)+' with weights')
            print (topic[:num_terms] if num_terms else topic)
        else:
            print ('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print (tw[:num_terms] if num_terms else tw)
        print()
        
def basic_lda(total_topics,corpus,dictionary,docs,score=False):
    #total_topics = 15
    print('Training for {} documents ......'.format(len(corpus)))
    
    lda = LdaModel(corpus = corpus,
                              id2word = dictionary,
                              num_topics = total_topics,
                              random_state = 2,
                              alpha='auto',
                              eta = 'auto')#,
                              #workers = 20) #
                              #iterations = 1000,
    # Compute Coherence Score
    if score:
        print('calculating coherence socre for {} documents ......'.format(len(docs)))
        coherence_model_lda = CoherenceModel(model=lda, texts=docs, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

        return lda,coherence_lda
    
    return lda

def seeded_lda(total_topics,corpus,dictionary,docs,seed_topic_list, boost, score=False):
    print('Modify beta prior ......')
    _model = LdaModel(corpus = corpus_bow, id2word = dictionary,random_state = 2,alpha='auto',num_topics = total_topics,iterations=0)
    beta_matrix = _model.expElogbeta
    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            try:
                w_id = dictionary.token2id[word]
                beta_matrix[t_id,w_id] = boost
                print('{} : {} : {}'.format(t_id,w_id,word))
            except:
                continue
    print('Training for {} documents ......'.format(len(corpus)))
    seed_model = LdaModel(corpus = corpus_bow,
                                  id2word = dictionary,
                                  num_topics = total_topics,
                                  eta = beta_matrix,
                                  random_state=2)
    # Compute Coherence Score
    if score:
        print('calculating coherence socre for {} documents ......'.format(len(docs)))
        coherence_model_lda = CoherenceModel(model=seed_model, texts=docs, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

        return seed_model,coherence_lda
    
    return seed_model
#%%
if __name__=="__main__":
    save = False  ## save gensim objects, corpus, dictionary, and lda model
    docs,dictionary,corpus_bow,corpus_tfidf = prepare_data(save=save)
    corpus_bow = [c for c in corpus_bow if len(c)>0]
    #%%
    n_topics = 25
    boost = 1000
    seed_topic_list = [['mpm','MPM','CFM','cfm','ltv','LTC','DSTI','dsti','lcr','LCR',
                        'capital_buffer','macroprudential','capital_flow','prudential'],
                        ['population','ageing','pension','productivity','migration','migrat']]
        
    seed_model = seeded_lda(n_topics,corpus_bow,dictionary,docs,seed_topic_list, boost, score=False)
    
    #%%
    
    coherence_model_lda = CoherenceModel(model=seed_model, texts=docs, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(coherence_lda)
    #%%
    
    #print_topics_gensim(topic_model=seed_model,
    #                   total_topics = n_topics,
    #                   num_terms=20,
    #                   display_weights=True)