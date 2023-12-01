# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 22:43:33 2019

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:27:24 2019

@author: dell
"""

import sys
import argparse
import json
import numpy as np
from LDA import lda_model, corp_dict
import random as rd
from gensim.models import CoherenceModel
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

if __name__=='__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument("-k","--k",type = int,default = 5)#topic number
     #parser.add_argument("-tf","--tfidf",action="store_false")
     parser.add_argument('--tfidf', dest='tf_idf', action='store_true')
     parser.add_argument('--no-tfidf', dest='tf_idf', action='store_false')
     parser.set_defaults(tf_idf=True)
     #parser.add_argument("-tr","--train",action="store_false")# whether or not select model
     parser.add_argument('--train', dest='train', action='store_true')
     parser.add_argument('--no-train', dest='train', action='store_false')
     parser.set_defaults(train_model=True)
     #parser.add_argument("-ts",'--tsne',action="store_true", default = False)# whether or not use tsne 
     
     
     '''
     relative loose parameters #for lda model (gensim)
     '''
     parser.add_argument("-cksz","--chunksize",type = int,default = 32)
     parser.add_argument("-ps","--passes",type = int,default = 10)
     parser.add_argument("-ite","--iteration",type = int,default = 5)
     parser.add_argument("-db","--dictionary_below",type = int,default = 10)
     parser.add_argument("-da","--dictionary_above",type = float,default = 0.9)
     #parser.add_argument("-wks","--workers",type = int,default = 3) #parrallized
     parser.add_argument("-al","--alpha",type = str,default = 'asymmetric')
     parser.add_argument("-dc","--decay",type = float,default = 0.5)
     
     args = parser.parse_args()
     #print(args.k,args.train,args.tf_idf)
     print('Get dic and corpus!')
     cp_dic = corp_dict(tf_idf = args.tf_idf,dic_below = args.dictionary_below,dic_above = args.dictionary_above)
     corpus = cp_dic.corpus
     dictionary = cp_dic.dictionary     
     processed_docs = cp_dic.processed_docs
     inp = open('test.json','rb')
     data = pd.DataFrame(json.load(inp))
     inp.close()
     
     
     def train_model():
         print('choose topics!')
         top_lst = list(range(2,11)) + list(range(12,20,2)) + list(range(20,51,3)) + list(range(55,101,5))
         tfidf_v = [True,False]
         min_prep = 10000000#init
         min_k=-1
         min_tfidf = None
         #record to plot
         tfidf_prep_v = []
         prep_v = []
         for tf_idf in tfidf_v:
             for k in top_lst:
                 print(k)
                 train_idx = rd.sample(range(len(corpus)),int(0.9*len(corpus)))
                 test_idx = list(set(range(len(corpus))).difference(set(train_idx)))
                 train_corp = cp_dic.get_extra(np.array(processed_docs)[train_idx],tf_idf)
                 test_corp = cp_dic.get_extra(np.array(processed_docs)[test_idx],tf_idf)
                 
                 _lda_model = lda_model(topic_num=k,corpus=train_corp,dictionary=dictionary,ite=args.iteration,ps=args.passes,
                               ck_size=args.chunksize,alpha=args.alpha,tf_idf=tf_idf,decay = args.decay)
                 cur_prep = _lda_model.get_prep(test_corp)
                 if cur_prep < min_prep:
                     min_k,min_tfidf = k,tf_idf
                     min_prep = cur_prep
                 print(min_k,min_tfidf)
                 print('topic:{0}--tf_idf{1}->prep:{2}'.format(k,tf_idf,cur_prep))
#         _lda_model = lda_model(topic_num=min_k,corpus=corpus,dictionary=dictionary,ite=args.iteration,ps=args.passes,
#                               ck_size=args.chunksize,alpha=args.alpha,tf_idf=min_tfidf,decay = args.decay)
                 if tf_idf:
                     tfidf_prep_v.append(cur_prep)
                 else:
                     prep_v.append(cur_prep)
         #_lda_model.save_model()
         print('min_k:{0},min_tfidf:{1},min_prep:{2}'.format(min_k,min_tfidf,min_prep))
         #return _lda_model
         
         #plot 
         
        #设置图形大小
        #save file for safety
         outp1 = open("perplexity_tfidf.json", 'w', encoding="utf-8")
         outp1.write(json.dumps(tfidf_prep_v, indent=4, ensure_ascii=False))
         outp1.close()
         outp2 = open("perplexity.json", 'w', encoding="utf-8")
         outp2.write(json.dumps(prep_v, indent=4, ensure_ascii=False))
         outp2.close()
         
         matplotlib.use('pdf') #prevent linux server cmd error
         plt.figure(figsize=(20,8),dpi=80)
        # color可以百度颜色代码
         plt.plot(top_lst,tfidf_prep_v,label="tf_idf",color="#F08080")
         plt.plot(top_lst,prep_v,label="no-tf_idf",color="#DB7093",linestyle="--")
         plt.xlabel('number of topics')
         plt.ylabel('log_perplexity')
         plt.grid(alpha=0.4,linestyle=':')
        #添加图例，prop指定图例的字体, loc参数可以查看源码
         plt.legend(loc="upper left")
         plt.savefig('train.jpg')


         
         
     
     if args.train:
#         _lda_model = train_model()
#         _lda_model.tsne_vis(data)
#         _lda_model.lda_vis(corpus=corpus,dictionary=dictionary)
         train_model() #only plot but not directly get the most suitable model/see it from eye
         
     else:
         _lda_model = lda_model(topic_num=args.k,corpus=corpus,dictionary=dictionary,ite=args.iteration,ps=args.passes,
                               ck_size=args.chunksize,alpha=args.alpha,tf_idf=args.tf_idf,decay = args.decay)
         #_lda_model.show_lda()
         #_lda_model.tsne_vis(data)
         _lda_model.lda_vis()
         
     

        
      
        
        
     
        
        
        
         
         