# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:06:33 2018

@author: chuang
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
    dictionary.filter_extremes(no_below=5,no_above=0.5, keep_n=10000)
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
                              alpha='auto',
                              eta = 'auto',
                              random_state = 2)#,
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

def mallet_lda(model_path,total_topics,corpus,dictionary,docs,score=False):
    """
    https://radimrehurek.com/gensim/models/wrappers/ldamallet.html
    sudo apt-get install default-jdk
    sudo apt-get install ant
    git clone git@github.com:mimno/Mallet.git
    cd Mallet/
    ant
    
    we don't have those packages in server environment
    """
    lda = gensim.models.wrappers.LdaMallet(model_path, corpus=corpus, num_topics=total_topics, id2word=dictionary)
    if score:
        print('calculating coherence socre for {} documents ......'.format(len(docs)))
        coherence_model = CoherenceModel(model=lda, texts=docs, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print('\nCoherence Score: ', coherence_score)
        return lda,coherence_score
    
def hdp(corpus,dictionary,docs,score=False):
    print('Traiing for {} documents ......'.format(len(corpus)))
    hdpmodel = HdpModel(corpus = corpus,id2word = dictionary)
    if score:
        print('calculating coherence socre for {} documents ......'.format(len(docs)))
        coherence_model = CoherenceModel(model=hdpmodel, texts=docs, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print('\nCoherence Score: ', coherence_score)
        return hdpmodel,coherence_score
    return hdpmodel
    
def lsi(total_topics, corpus,dictionary,docs,score=False):
    print('Traiing for {} documents ......'.format(len(corpus)))
    lsimodel = LsiModel(corpus = corpus,id2word = dictionary,num_topics=total_topics)
    if score:
        print('calculating coherence socre for {} documents ......'.format(len(docs)))
        coherence_model = CoherenceModel(model=lsimodel, texts=docs, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print('\nCoherence Score: ', coherence_score)
        return lsimodel,coherence_score
    return lsimodel

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


def fine_tune_lda(corpus, dictionary, texts, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    n_topics : numbmber of topics
    """
    coherence_values = []
    model_list = []
    n_topics = []
    for num_topics in range(start, limit, step):
        print('\nTraing with n_topics = {}, training sample = {}.'.format(num_topics,len(corpus)))
        model = LdaModel(corpus = corpus,
                          id2word = dictionary,
                          random_state = 2,
                          alpha='auto',
                          eta = 'auto',
                          num_topics = num_topics)#
                          #distributed = True)  # alpha='auto' is not implenented in distributed lda
        model_list.append(model)
        print('Calculating coherence score based on {} samples.'.format(len(texts)))
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        n_topics.append(num_topics)
        print("{}: {}".format(num_topics,coherence_values[-1]))
        

    return model_list, coherence_values,n_topics

#%%
if __name__== "__main__":
    
    save = True  ## save gensim objects, corpus, dictionary, and lda model
    mode = 'all'
    docs,dictionary,corpus_bow,corpus_tfidf = prepare_data(save=save)
    corpus_bow = [c for c in corpus_bow if len(c)>0]
    
    if mode == 'lda' or mode=='all':
        n_topics = 25
        model, score = basic_lda(total_topics=n_topics,corpus=corpus_bow,dictionary=dictionary,docs=docs,score=True)
        print(score)
        print_topics_gensim(topic_model=model,
                           total_topics = n_topics,
                           num_terms=20,
                           display_weights=True)
    if mode =='seed_lda' or mode=='all':
        n_topics = 25
        boost = 1000
        seed_topic_list = [['mpm','MPM','CFM','cfm','ltv','LTC','DSTI','dsti','lcr','LCR',
                            'capital_buffer','macroprudential','capital_flow','prudential'],
                            ['population','ageing','pension','productivity','migration','migrat']]
            
        seed_model = seeded_lda(n_topics,corpus_bow,dictionary,docs,seed_topic_list, boost, score=False)
        ## for some reason keeps buging out when calculating coherence score 
        
        print_topics_gensim(topic_model=seed_model,
                           total_topics = n_topics,
                           num_terms=20,
                           display_weights=True)
    
    if mode == 'fine_tune' or mode =='all':
        
        model_list, coherence_values,n_topics = fine_tune_lda(dictionary=dictionary, corpus=corpus_bow,
                                                            texts=docs, start=15, limit=35, step=1)
        
        best_model = model_list[np.argmax(coherence_values)]
        best_topic_n = best_model.get_topics().shape[0]
        
        plt.plot(n_topics, coherence_values)
        plt.show()
        
        print_topics_gensim(topic_model=best_model,
                       total_topics = best_topic_n,
                       num_terms=10,
                       display_weights=True)
        if save:
            lda_model_filepath = '../data/lda_res'
            best_model.save(lda_model_filepath)

    
    
    
    
    
    