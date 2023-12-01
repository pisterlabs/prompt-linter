# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:30:41 2019

@author: 
    using to select event from semantic lines and visiuize LDA to check consistency
"""
import os
#import sys
import argparse
import json
import numpy as np
from LDA import lda_model, corp_dict
#import random as rd
#from gensim.models import CoherenceModel
import gensim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
#from datetime import Datetime
import datetime
import matplotlib as mpl

if __name__=='__main__':
    print('begin')
    parser = argparse.ArgumentParser()
    parser.add_argument("-k","--k",type = int,default = 8)#topic number
    parser.add_argument('--tfidf', dest='tf_idf', action='store_true')
    parser.add_argument('--no-tfidf', dest='tf_idf', action='store_false')
    parser.set_defaults(tf_idf=True)
    
    parser.add_argument("-st_time",'--start_time',type = str,default = None)
    parser.add_argument('-ed_time','--end_time',type = str,default = None)
    parser.add_argument('-granu','--granularity',type = str,default = 'day')
    
    parser.add_argument("-cksz","--chunksize",type = int,default = 32)
    parser.add_argument("-ps","--passes",type = int,default = 10)
    parser.add_argument("-ite","--iteration",type = int,default = 5)
    parser.add_argument("-db","--dictionary_below",type = int,default = 15)
    parser.add_argument("-da","--dictionary_above",type = float,default = 0.9)
    parser.add_argument("-al","--alpha",type = str,default = 'asymmetric')
    parser.add_argument("-dc","--decay",type = float,default = 0.5)
    
    args = parser.parse_args()
    
#    def process_data(path = 'test.json'):
#        inp = open(path,'rb')
#        data = json.load(inp)
#        data = pd.DataFrame(data)
#        data = data.fillna('') #na request
#        inp.close()
#        data['time'] = pd.to_datetime(data.time.values)
#        #sort time 1st
#        data = data.sort_index(by = 'time',ascending = True)
#        data = data.drop_duplicates(subset=['passage'], keep=False)
        
    inp = open('test.json','rb')
    data = json.load(inp)
    data = pd.DataFrame(data)
    print(data.head())
    data['time'] = pd.to_datetime(data.time.values)
    #get data prepared and LDA model ready
    
    labels = data['label'].values
    passages = data['passage'].values
    headlines = data['headline'].values
    str_time = data['time'].values
    
    #get semantic-scaled
    
    #change neg's sign
    semantic_value = data['semantic_value'].values
    semantic_value = np.array([np.array(x) for x in semantic_value])
    semantic_arr = semantic_value.max(1) #get semantic value ready
    neg_idx = np.where(labels==0)#0 represent neg, 1 represent pos
    pos_idx = np.where(labels==1)
    semantic_arr[neg_idx] = -semantic_arr[neg_idx]#get full representative semantics
    data['semantic_arr'] = semantic_arr
    
    #scale
    #scale the data so the plot be more obvious / 分别对pos和neg部分scale，拉开差距

    neg_semantic = semantic_arr[neg_idx].reshape(-1, 1)
    pos_semantic = semantic_arr[pos_idx].reshape(-1,1)
    pos_scaler = preprocessing.StandardScaler().fit(pos_semantic)
    neg_scaler = preprocessing.StandardScaler().fit(neg_semantic)
    
    pos_semantic = pos_scaler.transform(pos_semantic)
    pos_semantic = np.array([float(x) for x in pos_semantic])
    neg_semantic = neg_scaler.transform(neg_semantic)
    neg_semantic = np.array([float(x) for x in neg_semantic])

    
    scale_semantic = np.zeros(len(semantic_arr))
    scale_semantic[neg_idx] = neg_semantic
    scale_semantic[pos_idx] = pos_semantic
    data['scale_semantic'] = scale_semantic
    
    str_time = data['time'].values
    str_time = [str(x).split('.')[0] for x in str_time]
    #print(str_time[:100])
    myFormat = "%Y-%m-%d %H:%M"
    datetime_arr = [datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(myFormat) for x in str_time]
    datetime_arr = [datetime.datetime.strptime(x,myFormat) for x in datetime_arr] #change to datetime obj
    all_time_arr = pd.to_datetime(datetime_arr)
    
    #set granu
    gran = args.granularity
    if gran=='day':
        time_index = [x.replace(hour = 0,minute = 0) for x in datetime_arr]#according to granularity choose the suitable time index
        time_index = pd.to_datetime(time_index)
    elif gran=='hour':
        time_index = [x.replace(minute = 0) for x in datetime_arr]#according to granularity choose the suitable time index
        time_index = pd.to_datetime(time_index)
    #groupby data and get every day's semantic value
    st_time,ed_time = args.start_time,args.end_time
    _idx = (time_index<=ed_time)&(time_index>=st_time)
    
    #groupby
    test_data = data.loc[_idx,] #using franuliarity
    test_data['time_index'] = time_index[_idx,]
    tmp = test_data.groupby('time_index').mean()  #get granu avg
    
    #plot
    _xlabel = 'time' + 'granularity: '+gran
    _path = os.getcwd() + '\\semantic_plot\\' + args.start_time + '--' + args.end_time + '--' + gran + '.jpg'
    
    mpl.rcParams['agg.path.chunksize'] = 10000
    mpl.use('pdf')#for cmd command
    plt.figure(figsize=(20,10),dpi=300)
#data.set_index('time')
    plt.plot(tmp['scale_semantic'])
    
    plt.xlabel(_xlabel)
    plt.ylabel('semantic_value')
    
    plt.savefig(_path)
    
    
    #select abnormal text and using LDA-tsne VIS for comparison
    sel_day = str(np.argmin(tmp['semantic_arr']))
    if gran=='day':
        _from = datetime.datetime.strptime(sel_day, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        _from = datetime.datetime.strptime(_from,'%Y-%m-%d')
        _to = _from + datetime.timedelta(days=1)
    elif gran=='hour':
        _from = datetime.datetime.srtptime(sel_day,'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H')
        _from = datetime.datetime.strptime(_from,'%Y-%m-%d %H')
        _to = _from + datetime.timedelta(hours = 1)
        
    sel_idx = (all_time_arr<=_to)&(all_time_arr>=_from)
    
    
    ####load lda_model
    #load corpus and dictionary 1st
    cp_dic = corp_dict(tf_idf = args.tf_idf,dic_below = args.dictionary_below,dic_above = args.dictionary_above)
    corpus = cp_dic.corpus
    dictionary = cp_dic.dictionary
    print('Begin LDA Modeling')
    _lda_model = lda_model(topic_num=args.k,corpus=corpus,dictionary=dictionary,ite=args.iteration,ps=args.passes,
                               ck_size=args.chunksize,alpha=args.alpha,tf_idf=args.tf_idf,decay = args.decay,path = 'lda_model8-tf_idf')
    _lda_model.save_model()
    print(sel_day)
    #_lda_model.tsne_vis(data,time_index = sel_idx) #select index
    _lda_model.lda_vis()
    #_lda_model.lda_vis(sel_idx)
    #_lda_model.tsne_vis(data,time_index = sel_idx)
    #_lda_model.tsne_vis(data)


    
