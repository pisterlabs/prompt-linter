#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:19:18 2020

@author: olivergiesecke
"""

import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
import itertools  
import os
import gensim
from gensim import corpora, models
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import re
import seaborn as sns
from gensim.models.coherencemodel import CoherenceModel

# Define functions for the LDA 

    # Calculate Hellinger distance which is between 0 and 1.
def hellinger(x,y,col_topics):
    array1 = x[col_topics].values[0]
    array2 = y[col_topics].values[0]
    helpvar = np.sqrt(1 - np.sum(np.sqrt(np.multiply(array1,array2))))
    return helpvar

def extract_token(sentence):
    return simple_preprocess(str(sentence), deacc=True)

def count_mostcommon(myList):
    counts = {}
    for item in myList:
        counts[item] = counts.get(item,0) + 1
    return counts

def remove_stopwords(words,stopwords):
    nostopwords=[ word for word in words if word not in stopwords]
    return nostopwords

def do_stemming(words):
    p_stemmer = PorterStemmer()
    stemmed_words = [p_stemmer.stem(i) for i in words]
    return stemmed_words 

def create_wordcounts(data,stop_words):
        # List the 100 most common terms
    tot_token=[]
    for row_index,row in data.iterrows():
        tot_token.extend(row['parsed'])
    print('The corpus has %d token' % len(tot_token) )
    counts=count_mostcommon(tot_token)
    sorted_counts={k: v for k, v in sorted(counts.items(), key=lambda item: item[1],reverse=True)}
    
    N = 100
    out = dict(itertools.islice(sorted_counts.items(), N))  
    words100 = [(k, v) for k, v in out.items()]
    print(f'This are the most common {N} tokens without removing stopwords:')
    print(words100)
    
    # Remove stopwords
    wo_stopwords = remove_stopwords(tot_token,stop_words)
    
    wo_counts=count_mostcommon(wo_stopwords)
    sorted_wo_counts={k: v for k, v in sorted(wo_counts.items(), key=lambda item: item[1],reverse=True)}
    
    # Do stemming
    
    wo_stem_word = do_stemming(wo_stopwords)
    
    wo_counts=count_mostcommon(wo_stem_word)
    sorted_wo_counts={k: v for k, v in sorted(wo_counts.items(), key=lambda item: item[1],reverse=True)}
    
    out = dict(itertools.islice(sorted_wo_counts.items(), N))  
    wo_words100 = [(k, v) for k, v in out.items()]
    print(f'This are the most common {N} tokens with removing stopwords and stemming:')
    print(wo_words100)

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i texts..." % (M.shape[0]))
    
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    svd.fit(M)
    M_reduced=svd.transform(M)
 
    print("Done.")
    return M_reduced

def extract_vectors(ldamodel,num_topics,corpus): 
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        emptylist=[]
        for k in range(num_topics):          
            emptylist.append(0)
            for j, (topic_num, prop_topic) in enumerate(row):    
                if k==topic_num:
                    emptylist[-1]=(round(prop_topic,4))
        sent_topics_df = sent_topics_df.append(pd.Series(emptylist), ignore_index=True)
    
    coln = sent_topics_df.columns
    coln = ['topic_%s'%c for c in coln]
    sent_topics_df.columns=coln
    return sent_topics_df

def output_plot(date,data):
    plt.figure(figsize=(12,7))
    for i, row in data.iterrows():
        if re.search("^alt[a-d]", row["speaker"]):
            plt.scatter(row['PCI1'],row['PCI2'], edgecolors='k', c='b')
            plt.text(row['PCI1'],row['PCI2'], row["speaker"])
            
        else:
            plt.scatter(row['PCI1'],row['PCI2'], edgecolors='k', c='r')
            plt.text(row['PCI1'],row['PCI2'], row["speaker"])
    plt.title('Preference Vector 2D for %s' %date )
    plt.savefig('../output/fig_pref_vec_%s.pdf'%date)

def create_distance(date,data,col_topics):
    emptylist=[]
    for speaker in data.loc[(data['date']==date) & (data['votingmember']==1),"speaker"].to_list():
        dic={'date':date,'speaker':speaker}
        z=data[(data['date']==date) & (data['speaker']==speaker)]
        #print(z)
        for alt in ['a','b','c','d','e']:
            zz=data[(data['date']==date) & (data['speaker']=='alt'+alt)]
            #print(zz)
            #print(f"This is alternative {alt} and topic distribution\n"  )
            #print(zz)
            if zz.index.empty or z.index.empty:
                pass
            else:
                dist=hellinger(z,zz,col_topics)
                dic.update({'alt'+alt:dist})
                #print(dic)
        emptylist.append(dic)        
        
    distances = pd.DataFrame(emptylist)
    return distances

def add_bigrams(texts,bi_gram_mostcommon):
    for idx,text in enumerate(texts):
        bi_grams = list(ngrams(text, 2)) 
        bi_grams_con = ["_".join(bi) for bi in bi_grams]
        for ele in bi_grams_con:
            if ele in bi_gram_mostcommon:
               texts[idx].append(ele)
    return texts

def add_trigrams(texts,tri_gram_mostcommon):
    for idx,text in enumerate(texts):
        tri_grams = list(ngrams(text, 3)) 
        tri_grams_con = ["_".join(tri) for tri in tri_grams]
        for ele in tri_grams_con:
            if ele in tri_gram_mostcommon:
                texts[idx].append(ele)            
    return texts

def get_tdidf(tokens,unique_tokens,texts):

    n_v = np.zeros(len(unique_tokens))
    d_v = np.zeros(len(unique_tokens))
    d = len(texts)
    for idx,term in enumerate(unique_tokens):
        n_v[idx] =  tokens.count(term) # total count of word
        counter = 0
        for text in texts:
            if term in text:
                counter+=1
        d_v[idx] = counter
    
    tf = 1 + np.log(n_v)
    idf = np.log(d / d_v )
    
    tf_idf = np.multiply(tf,idf)
    return tf_idf

def trim_texts(tf_idf,unique_tokens,texts,nn):
    indices = tf_idf.argsort()[-nn:][::-1]
    
    word_arr = np.asarray(unique_tokens)
    word_top= list(word_arr[indices])
    
    newtexts =[]
    for text in texts:
        newelement=[]
        for word in text:
            if word in word_top:
                newelement.append(word)
                
        newtexts.append(newelement)
     
    return newtexts

def draw_heatmap(x,n_words,params, pmin = 0, pmax = 1):
    plt.rcParams['text.usetex']=False
    plt.rcParams['text.latex.unicode']=False

    emptylist = []
    for topic in x:
        dicc={'topic':topic[0]+1}
        for idx,word in enumerate(topic[1]):
            dicc.update({f"word{idx+1}":word[1]} ) 
      
        emptylist.append(dicc)
        
    topics_df = pd.DataFrame(emptylist)
    topics_df=topics_df.set_index('topic')
    
    
    wordlabel=[item[0] for tp in x for item in tp[1]]
    labels = np.asarray(wordlabel).reshape(int(params[0]),n_words)
    
    fig,ax=plt.subplots(figsize=(20,7))
    fig.suptitle(f'Topic Distribution')
    sns.heatmap(topics_df,annot=labels,cmap="Blues",fmt="",ax=ax, vmin=pmin, vmax=pmax)
    plt.savefig('../output/fig_topicheatmap.pdf')
    

def explore_parameterspace(totaln_words,corpus,dictionary,rnd_state,texts,alpha_v,eta_v,topic_v):
    
    alpha = np.repeat(np.repeat(alpha_v,len(eta_v)),len(topic_v))
    eta = np.tile(eta_v,len(alpha_v) * len(topic_v))
    topic = np.tile(np.repeat(topic_v,len(eta_v)),len(eta_v))
    logperplexity = np.zeros(len(alpha_v) * len(topic_v) *len(eta_v))
    coh_score = np.zeros(len(alpha_v) * len(topic_v) *len(eta_v))
    for i in range(len(alpha_v) * len(topic_v) *len(eta_v) ):
        num_topics = topic[i]
        eta_p = eta[i]
        alpha_p = alpha[i]
        ldamodel = models.ldamodel.LdaModel(corpus, num_topics, id2word = dictionary, passes=30,eta=eta_p ,alpha = alpha_p, random_state=rnd_state)
        logperplexity[i] = ldamodel.log_perplexity(corpus, total_docs=None)
        cm = CoherenceModel(model=ldamodel, corpus=corpus, texts = texts, coherence='c_uci') # this is the pointwise mutual info measure.
        coh_score[i] = cm.get_coherence()  # get coherence value
        print(f"Number of topics {num_topics}, \t eta: {eta_p}, \t alpha: {alpha_p}; \t Coherence: {coh_score[i]}; \t Peplexity: {logperplexity[i]}")
    
    dataarray = np.array((np.arange(len(alpha_v) * len(topic_v) *len(eta_v) ) +1, topic,alpha,eta,coh_score , logperplexity )).T
    models_df = pd.DataFrame(data=dataarray ,columns=['model','# topics', 'alpha','eta','coherence score (PMI)','perplexity'])
    models_df['model']=models_df['model'].astype(int)
    models_df['# topics']=models_df['# topics'].astype(int)
    return models_df


def plot_parameterspace(data):
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('eta')
    ax.set_ylabel('alpha')
    ax.set_zlabel('\# topics')
    
    x = data['eta']
    y = data['alpha']
    z = data['# topics']
    c = data['coherence score (PMI)']
    
    img = ax.scatter(x, y, z, c=c, cmap="Blues", vmin=-8, vmax=-4,label='Coherence Score')
    ax.legend()
    fig.colorbar(img)
    plt.savefig("../output/fig_parameterspace.pdf")
    
    
def explore_numberoftopics(totaln_words,corpus,dictionary,texts,rnd_state,eta_p,alpha_p):
    nn = 10
    coh_score = np.zeros(nn)
    logperplexity = np.zeros(nn)
    n_topics= np.zeros(nn)
    for i in range(nn):
        num_topics = 5 * (i+1)    
        n_topics[i] = num_topics
        ldamodel = models.ldamodel.LdaModel(corpus, num_topics, id2word = dictionary, passes=30,eta=eta_p ,alpha = alpha_p, random_state=rnd_state)
        logperplexity[i] = ldamodel.log_perplexity(corpus, total_docs=None)
        cm = CoherenceModel(model=ldamodel, corpus=corpus, texts = texts, coherence='c_uci') # this is the pointwise mutual info measure.
        coh_score[i] = cm.get_coherence()  # get coherence value
        print(f"Number of topics is: {num_topics}; \t coherence score is: {coh_score[i]}: \t logperplexity is: {logperplexity[i]}")
    
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(111)
    ax1.plot(n_topics,coh_score,color = 'tab:red' )
    ax1.set_xlabel('number of topics',size=20)
    ax1.set_ylabel('coherence score',color = 'tab:red', size=20)
    
    ax2 = fig.add_subplot(111,sharex=ax1, frameon=False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.plot(n_topics,logperplexity,color = 'tab:blue' )
    ax2.set_ylabel('log perplexity',color = 'tab:blue', size=20)

    plt.savefig("../output/fig_numberoftopics.pdf")
    

def plot_wordlist(counter,n_topwords ,n_percolumns,filename,columnnames=['#','term','count']):

    n_columns = n_topwords // n_percolumns
    if n_topwords % n_percolumns != 0:
        n_columns += 1    
    
    #print(n_columns)
    df_all=pd.DataFrame()
    columns_old = 1
    newelement = []
    for idx,element in enumerate(counter):
        #print(idx)
        columns = (idx) // n_percolumns + 1
        #print(columns)
        if columns_old == columns:
            element = tuple([idx+1]) +  element 
            newelement.append(element)
        else:
            df=pd.DataFrame(newelement,columns=columnnames)
            df_all = pd.concat([df_all,df],axis=1)
            columns_old = columns
            newelement = []
            element = tuple([idx+1]) +  element 
            newelement.append(element)
    
    df=pd.DataFrame(newelement,columns=columnnames)
    df_all = pd.concat([df_all,df],axis=1)
    df_all = df_all.replace(np.nan, '', regex=True)
    df_all.to_latex(filename,index=False)   
    return df_all

def output_number(number,filename,dec=2):
    with open(filename, "w") as f:   
        print(f"{number :.{dec}f}")
        f.write(f"{number :.{dec}f}") 
