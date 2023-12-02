# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:30:43 2018

@author: vprayagala2
Build Model
Write function for each of machine learning experimentation
"""
#%%
#Get the logger
import logging
from Source.Config import LoadConfiguration as LC
from Source.DataHandler import PrepareData as PD

#import os
#import numpy as np
import pandas as pd
#from sklearn.metrics import silhouette_score
# Gensim
import gensim
from gensim.summarization import keywords
import gensim.corpora as corpora
#from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
#import spacy
# Plotting tools
#import pyLDAvis
#import pyLDAvis.gensim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
#%%
#Load Configurtion and get logger
LC.load_config_file()
logger=logging.getLogger(LC.getParmValue('LogSetup/Log_Name'))
#%%
#Define Functions 
def cluster_texts_kmeans(texts, clusters=5,true_k=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(#max_df=0.5,
                                 #min_df=0.1,
                                 #lowercase=True)
                                    )
    tfidf_model = vectorizer.fit_transform([word for word in texts])
    
    #Fit different cluster and pick the optimal cluster size
    df_clust=pd.DataFrame()
    for i in range(2,clusters+1):
        #Build model
        logger.info("Building Kmean with {} cluster".format(i))
        km_model = KMeans(n_clusters=i,random_state=7)
        km_model.fit(tfidf_model)
        #labels=km_model.labels_
        #score=silhouette_score(tfidf_model, labels, metric='euclidean')
        score=km_model.inertia_
        logger.info("K-Means Score:{}".format(score))
        df_clust=df_clust.append({"num_clusters":i,"score":score},ignore_index=True)
    
    plt.figure()
    plt.plot(df_clust["num_clusters"],df_clust["score"])
    plt.savefig("kmeans_elbow.png")
    #clustering = collections.defaultdict(list)
    #for idx, label in enumerate(km_model.labels_):
    #    clustering[label].append(idx)
   
    km=KMeans(n_clusters=true_k,random_state=77)
    km.fit(tfidf_model)
    kmeans_clust=pd.DataFrame()
    
    logger.info("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    
    for i in range(true_k):
        term_list=[]
        logger.info("Cluster %d:\n" % i)
        for ind in order_centroids[i, :15]:
            logger.info(' %s' % terms[ind])
            term_list.append(terms[ind])
        kmeans_clust=kmeans_clust.append({"Cluster_Num":i,"Top_Terms":term_list},\
                                         ignore_index=True) 

    return km,kmeans_clust

def topic_modeling_lda(texts, max_topics=5,true_topics=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    #Explore Topic Modeling
    ## python3 -m spacy download en
    
    # Create Dictionary
    bigram = gensim.models.Phrases(texts)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    data_bigrams=[bigram_mod[sentence] for sentence in texts]
    data_cleaned = PD.lemmatization(data_bigrams,\
                                    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    #Dictionary of Word Mappings
    id2word = corpora.Dictionary(data_cleaned)
    # Term Document Frequency
    tdm = [id2word.doc2bow(word) for word in data_cleaned]
    
    df_result=pd.DataFrame()
    for i in range(2,max_topics+1):
        logger.info("Experimenting LDA with {} Topics".format(i))
        lda_model = gensim.models.ldamodel.LdaModel(corpus=tdm,
                                               id2word=id2word,
                                               num_topics=i, 
                                               random_state=7,
                                               update_every=1,
                                               chunksize=1,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True
                                               )
        # Compute Perplexity
        perplexity=lda_model.log_perplexity(tdm)
        logger.info('\nPerplexity: {}'.format(perplexity) )
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_cleaned, 
                                             dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        logger.info('\nCoherence Score: {}'.format(coherence_lda))
        df_result=df_result.append({"num_topics":i,
                                    "Perplexity":perplexity,
                                    "Coherence":coherence_lda
                                    },ignore_index=True)
    logger.info("Result of Experiment:{}".format(df_result))
    #Build the final topic model with true topics provided in configuration    
    lda_model = gensim.models.ldamodel.LdaModel(corpus=tdm,
                                       id2word=id2word,
                                       num_topics=true_topics, 
                                       random_state=7,
                                       update_every=1,
                                       chunksize=1,
                                       passes=10,
                                       alpha='auto',
                                       per_word_topics=True
                                       )    
    topics = lda_model.print_topics(num_topics=true_topics, num_words=15)
    logger.info("Topics:{}".format(topics))
    
    return lda_model,tdm,id2word

def extractTopKeywords(text,num_keywords=30):
    keyword_list =[]
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    #vectorizer = TfidfVectorizer()
    #tfidf_model = vectorizer.fit_transform([word for word in text])
    #terms = vectorizer.get_feature_names()

    #scores = tfidf_model.toarray().flatten().tolist()
    
    #data = list(zip(terms,scores))
    pos_tag=('NN','JJ','RB','VB','CD')
    cap=int(len(text.split()) * 0.2)
    if num_keywords >= cap:
        num_keywords = cap
        
    print("Extracting {} Keywords".format(num_keywords))
    data=keywords(text,scores=True,
                  pos_filter=pos_tag,
                  lemmatize=True,
                  words=num_keywords)
    sorted_data = sorted(data,key=lambda x: x[1],reverse=True)
    
    if len(sorted_data) > num_keywords:
        keyword_list = sorted_data[:num_keywords]
    else:
        keyword_list = sorted_data
        
    return keyword_list