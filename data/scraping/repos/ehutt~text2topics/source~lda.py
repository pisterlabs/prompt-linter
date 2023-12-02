#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:12:39 2019

@author: elizabethhutton
"""


from gensim.models import CoherenceModel, LdaModel
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
import math
import iterate 
import pandas as pd

class LDA() :
    
    def __init__(self,corpus,num_topics,num_iter):
        if type(num_topics) != int or type(num_iter) != int: 
            print('Expected integer values!')
            return
        lda = LdaModel
        self.topics = num_topics
        self.iter = num_iter
        self.coherence = 0
        self.model = lda(corpus.dtm, num_topics=num_topics, id2word = corpus.dictionary, passes = num_iter)
        return 
    
    def save_model(self,model_file):
        """Save LDA model to model_file path."""
        self.model.save(model_file)
        return
    
    def get_coherence_score(self,corpus):
        if type(corpus.tokens[0]) != list: 
            print('Expected documents as tokens!')
        coherencemodel = CoherenceModel(model=self.model, texts=corpus.tokens, dictionary=corpus.dictionary, coherence='c_v')
        coherence_score = coherencemodel.get_coherence()
        self.coherence = coherence_score
        return coherence_score
        
    def get_top_words(self):
        topics = self.model.show_topics(formatted=False)
        top_words = pd.DataFrame()
        for i in range(len(topics)):
            tuples = topics[i][1][:]
            words = list()
            for t in tuples:
                words.append(t[0])
            top_words['Cluster ' + str(i)] = words
        return top_words
    
    def iterate_lda(clean_corpus,elbow):
        #prepare for LDA
        clean_corpus, extra_stops = clean_corpus.remove_common_words(6)
        clean_corpus.make_dict() 
        clean_corpus.make_dtm()
        
        #iterate LDA over num topics
        model_list, coherence_scores = elbow.elbow_lda(clean_corpus, num_iter = 50)
        elbow.plot_coherence(coherence_scores)
        return    

    
def get_lda(clean_corpus,num_topics):
   
    clean_corpus, extra_stops = clean_corpus.remove_common_words(6)
    clean_corpus.make_dict() 
    clean_corpus.make_dtm()
    num_iter = 50
     
    lda_model = LDA(clean_corpus, num_topics, num_iter)
    top_words_lda = lda_model.get_top_words()
    
    return lda_model, top_words_lda